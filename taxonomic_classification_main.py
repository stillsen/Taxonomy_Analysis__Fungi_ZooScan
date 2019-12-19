import os, re, argparse
import pandas as pd
from multiprocessing import cpu_count
from matplotlib import pyplot as plt
import mxnet as mx

# load custom modules
from DataHandler import DataHandler
from DataPrep import DataPrep
from ModelHandler import ModelHandler

def parse_opts():
    parser = argparse.ArgumentParser(description='Taxonmic Rank Classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='fungi or zoo')
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
data_set = opts.data
print('processing %s'%data_set)
df = None
if data_set == 'fungi':
    path = "/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy"
    missing_values = ['', 'unknown', 'unclassified']
    taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']

    csv_path = os.path.join(path, 'im_merged.csv')
    df = pd.read_csv(csv_path, na_values=missing_values)
elif data_set == 'zoo':
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
    taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    csv_path = os.path.join(path, 'zoo_df.csv')
    df = pd.read_csv(csv_path, na_values=missing_values)

# PARAMETERS Training
epochs = 3
num_workers = cpu_count()
num_gpus = 1
per_device_batch_size = 1
batch_size = per_device_batch_size * max(num_gpus, 1)
metric = mx.metric.Accuracy()
# metric = mx.metric.F1()

# PARAMETERS Augmentation
jitter_param = 0.4
lighting_param = 0.1



print('NaNs in the label data set')
print(df.isnull().sum())


# taxonomic_groups = ['phylum']
# fig, ax = plt.subplots(2, 3, sharey='row')
# sns.set_context("paper")
fig = plt.figure(figsize=(15, 10))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
subplot = 1
for i, taxa in enumerate(taxonomic_groups):
    print('working in taxonomic group: %s' %taxa)

    print('prepraring data')
    DataPrep(taxa=taxa, path=path, type = data_set, df=df)

    print('loading image folder dataset')
    data_handler = DataHandler(path=path,
                               batch_size = batch_size,
                               num_workers=num_workers,
                               transform = True)
    classes = data_handler.classes

    print('loading model')
    model = ModelHandler(classes=classes,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         metrics=metric)
    ### load parameters if already trained, otherwise train
    model_loaded = False
    param_file = ''
    e = -1
    for file_name in os.listdir(path):
        if re.match('%s-%s-model_parameter'%(data_set,taxa), file_name):
            if int(file_name.split('-')[-1][0]) > e:
                e = int(file_name.split('-')[-1][0])
                param_file = os.path.join(path, file_name)
            model_loaded = True
    if not model_loaded: # train
        print('training model for %s-%s' %(data_set,taxa))
        model.train(train_iter=data_handler.train_data,
                    val_iter=data_handler.test_data,
                    epochs=epochs,
                    path=path,
                    taxa=taxa)
    else:
        model.net.load_parameters(param_file)
        print('loading %s' %param_file)


    x = list(data_handler.samples_per_class.keys())
    y = list(data_handler.samples_per_class.values())


    ax = fig.add_subplot(2, 3, i)
    title = 'Acc %s' %taxa
    plot_classification_acc(x=x,
                    y=y,
                    colors='b',
                    title=title,
                    axis=ax)
    # ax.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')

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

    val_names, val_accs = model.evaluate(model.net, data_handler.test_data, model.ctx)
    print('%s: %s' % (taxa, model.metric_str(val_names, val_accs)))


plt.tight_layout()
plt.show()
