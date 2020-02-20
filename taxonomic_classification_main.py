import os, re, argparse
import pandas as pd
from multiprocessing import cpu_count
from matplotlib import pyplot as plt
import mxnet as mx
from metrics import F1

# load custom modules
from DataHandler import DataHandler
from DataPrep import DataPrep
from ModelHandler import ModelHandler

def parse_opts():
    parser = argparse.ArgumentParser(description='Taxonmic Rank Classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='fun or zoo')
    opts = parser.parse_args()
    return opts

def plot_classification_acc(x, y, colors, title, axis=None):
    if axis == None:
        fig, ax = plt.subplots(figsize=(15, 6))
        class_plot = ax.bar(x, y, color=colors)
    else:
        class_plot = axis.bar(x, y, color=colors)
    # y ticks
    # for i, val in enumerate(y_fun):
    #     ax.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})
    # x ticks
    plt.gca().set_xticklabels(x, rotation=60, horizontalalignment='right')
    # title
    plt.title(title, fontsize=22)
    # labels
    plt.ylabel("#")
    # plt.xlabel('Taxon')

    # make room for x labels
    # plt.tight_layout()



opts = parse_opts()
dataset = opts.data
print('processing %s' % dataset)
if dataset == 'fun':
    # Parameters ---------
    path = "/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy"
    missing_values = ['', 'unknown', 'unclassified']
    taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    augment = 'transform'
    # --------------------

    csv_path = os.path.join(path, 'im_merged.csv')
    df = pd.read_csv(csv_path, na_values=missing_values)
elif dataset == 'zoo':
    # Parameters ---------
    path = "/home/stillsen/Documents/Data/ZooNet/ZooScanSet/imgs"
    missing_values = ['',
                          'artefact',
                          'bubble',
                          'detritus',
                          'seaweed',
                          'head',
                          'megalopa',
                          'Rhincalanidae',
                          'cirrus',
                          'metanauplii',
                          'cyphonaute',
                          'scale',
                          'Pyrosomatida',
                          'ephyra']
    # taxonomic_groups = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxonomic_groups = ['phylum', 'class', 'subclass', 'order', 'suborder', 'infraorder', 'family', 'genus', 'species']
    augment = 'resize'
    # --------------------

    csv_path = os.path.join(path, 'zoo_df.csv')
    df = pd.read_csv(csv_path, na_values=missing_values)

# PARAMETERS Training
epochs = 5
num_workers = cpu_count()
num_gpus = 1
# batch_size = 2006
per_device_batch_size = 1
batch_size = per_device_batch_size * max(num_gpus, 1)

#PARAMETERS Model
metric = mx.metric.Accuracy()
# metric = F1(average="micro")
# binary relevance approach -> ignores possible correlations
multilabel_lvl = 2
# multilabel/-class approach -> sigmoid in last layer
# multilabel_lvl = 2


# PARAMETERS Augmentation
jitter_param = 0.4
lighting_param = 0.1



print('NaNs in the label data set')
print(df.isnull().sum())

if multilabel_lvl == 1:
    fig = plt.figure(figsize=(15, 10))
    subplot = 1
    for i, taxa in enumerate(taxonomic_groups):
        print('------------------------------------------')
        print('working in taxonomic rank: %s' %taxa)

        print('prepraring data')
        data_prepper = DataPrep(rank=taxa, path=path, dataset= dataset, df=df, multilabel_lvl=multilabel_lvl, taxonomic_groups=taxonomic_groups)

        print('loading image folder dataset')
        data_handler = DataHandler(path=data_prepper.imagefolder_path,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   augment=augment)
        classes = data_handler.classes
        print('numer of classes: %s' % classes)

        print('loading model')
        model = ModelHandler(classes=classes,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             metrics=metric,
                             multi_label_lvl=multilabel_lvl)

        ### load parameters if already trained, otherwise train
        model_loaded = False
        param_file = ''
        e = -1
        for file_name in os.listdir(data_prepper.imagefolder_path):
            if re.match('%s-%s-'%(dataset, taxa), file_name):
                if int(file_name.split('-')[-1][0]) > e:
                    e = int(file_name.split('-')[-1][0])
                    param_file = os.path.join(data_prepper.imagefolder_path, file_name)
                model_loaded = True
        if not model_loaded: # train
            print('training model for %s-%s' % (dataset, taxa))
            model.train(train_iter=data_handler.train_data,
                        val_iter=data_handler.test_data,
                        epochs=epochs,
                        path=data_prepper.imagefolder_path,
                        dataset=dataset,
                        taxonomic_group=taxa)
        else:
            model.net.load_parameters(param_file)
            print('loading %s' %param_file)


        x = list(data_handler.samples_per_class.keys())
        y = list(data_handler.samples_per_class.values())


        ax = fig.add_subplot(2, 3, i+1)
        title = 'Acc %s' %taxa
        plot_classification_acc(x=x,
                        y=y,
                        colors='b',
                        title=title,
                        axis=ax)
        # ax.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
        #
        # if i < 3:
        #     ax[0][i].xaxis.set_visible(False)
        #     sns.barplot(x=x, y=y, color="b", ax=ax[0][i])
        #     ax[0][i].set_title(taxa)
        # else:
        #     ax[1][i - 3].xaxis.set_visible(False)
        #     sns.barplot(x=x,y=y, color="b", ax=ax[1][i - 3])
        #     ax[1][i - 3].set_title(taxa)


        for cl in data_handler.samples_per_class:
            print("not resampled %s --- %s: %d"%(taxa,cl,data_handler.samples_per_class[cl]))
            print("    resampled %s --- %s: %d" % (taxa, cl, data_handler.samples_per_class_normalized[cl]))

        val_names, val_accs = model.evaluate(model.net, data_handler.test_data, model.ctx, metric=metric)
        print('%s: %s' % (taxa, model.metric_str(val_names, val_accs)))


    plt.tight_layout()
    plt.show()
elif multilabel_lvl == 2:

    fig = plt.figure(figsize=(15, 10))

    print('working in multilabel taxonomic classification: sigmoid' )

    print('prepraring data')
    data_prepper = DataPrep(rank=None, path=path, dataset= dataset, df=df, multilabel_lvl=multilabel_lvl, taxonomic_groups=taxonomic_groups)

    print('loading image folder dataset')
    data_handler = DataHandler(path=data_prepper.imagefolder_path,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               augment=augment)
    classes = data_handler.classes
    print('numer of classes: %s'%classes)

    print('preparing model class')
    model = ModelHandler(classes=classes,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         metrics=metric,
                         multi_label_lvl=multilabel_lvl)

    ### load parameters if already trained, otherwise train
    model_loaded = False
    param_file = ''
    e = -1
    for file_name in os.listdir(path):
        if re.match('%s-m_lvl_2-'%(dataset), file_name):
            if int(file_name.split('-')[-1][0]) > e:
                e = int(file_name.split('-')[-1][0])
                param_file = os.path.join(path, file_name)
            model_loaded = True
    if not model_loaded: # train
        print('training model for %s multi lvl' % (dataset))
        model.train(train_iter=data_handler.train_data,
                    val_iter=data_handler.test_data,
                    epochs=epochs,
                    path=path,
                    dataset=dataset)
    else:
        model.net.load_parameters(param_file)
        print('loading %s' %param_file)


    x = list(data_handler.samples_per_class.keys())
    y = list(data_handler.samples_per_class.values())


    ax = fig.add_subplot(2, 3, i+1)
    title = 'Acc in ml.....strange yet'
    plot_classification_acc(x=x,
                    y=y,
                    colors='b',
                    title=title,
                    axis=ax)

    for cl in data_handler.samples_per_class:
        print("not resampled --- %s: %d"%(cl,data_handler.samples_per_class[cl]))
        print("    resampled --- %s: %d" % (cl, data_handler.samples_per_class_normalized[cl]))

    val_names, val_accs = model.evaluate(model.net, data_handler.test_data, model.ctx, metric=metric)
    print(' %s' % (model.metric_str(val_names, val_accs)))


    plt.tight_layout()
    plt.show()