import os, re
import pandas as pd
from multiprocessing import cpu_count
from matplotlib import pyplot as plt
from gluoncv.utils import makedirs
import mxnet as mx
from mxnet.metric import F1
from mxnet.metric import PCC

# load custom modules
from DataHandler import DataHandler
from DataPrep import DataPrep
from ModelHandler import ModelHandler


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


def load_or_train_model(model, dataset, path, param_folder_name, mode, epochs, ext_storage_path, app=''):
    param_folder_path = os.path.join(path, param_folder_name)
    if not os.path.exists(param_folder_path): makedirs(param_folder_path)
    ext_storage_folder_path = os.path.join(ext_storage_path, param_folder_name)
    if not os.path.exists(ext_storage_folder_path): makedirs(ext_storage_folder_path)

    param_file_name = '%s_%s%s.param' % (dataset, mode, app)
    app_file_name = '%s_%s%s.txt' % (dataset, mode, app)
    abs_param_file_name = os.path.join(param_folder_path, param_file_name)
    if os.path.exists(abs_param_file_name):
        print('\tloading %s' % param_file_name)
        model.net.load_parameters(abs_param_file_name)
    else:  # train
        print('\ttraining %s' % param_file_name)
        model.net = model.train(train_iter=data_handler.train_data,
                                val_iter=data_handler.test_data,
                                epochs=epochs,
                                param_folder_path=param_folder_path,
                                param_file_name=param_file_name,
                                ext_storage_path=ext_storage_path,
                                app_file_name=app_file_name)
    return model


# Metaparameters for file handling
path = "/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy"
ext_storage_path = '/media/stillsen/Elements SE/Data'
missing_values = ['', 'unknown', 'unclassified']
taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
augment = 'transform'
dataset = 'fun'
net_name = 'densenet169'

csv_path = os.path.join(path, 'im_merged.csv')
df = pd.read_csv(csv_path, na_values=missing_values)

print('NaNs in the label data set')
print(df.isnull().sum())

###################################################################################
#######################   Per Level Classifier  ###################################
###################################################################################
# PARAMETERS Training
multilabel_lvl = 1

epochs = 10
learning_rate = 0.001
# learning_rate = 0.0001
# learning_rate = 0.4
momentum = 0.8

param_folder_name = 'ParameterFiles_%s_e%i_lr%f_m%f' % (net_name, epochs, learning_rate, momentum)

num_workers = cpu_count()
num_gpus = 1
# batch_size = 2006
per_device_batch_size = 5
batch_size = per_device_batch_size * max(num_gpus, 1)

# PARAMETERS Model
# metric = mx.metric.Accuracy()
# metric = F1(average="micro")
# metric = F1()
metric = PCC()

# binary relevance approach -> ignores possible correlations

# PARAMETERS Augmentation
jitter_param = 0.4
lighting_param = 0.1

print("########################################")
print("Per Level Classifier  ")
# fig = plt.figure(figsize=(15, 10))
# subplot = 1

for i, taxa in enumerate(taxonomic_groups):
    print('working in taxonomic rank: %s' % taxa)

    print('\tmodule DataPrep.py: ... prepraring data')
    data_prepper = DataPrep(rank=taxa, path=path, dataset=dataset, df=df, multilabel_lvl=multilabel_lvl,
                            taxonomic_groups=taxonomic_groups)

    print('\tmodule DataHandler.py: ... loading image folder dataset and augmenting')
    data_handler = DataHandler(path=data_prepper.imagefolder_path,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               augment=augment)
    classes = data_handler.classes
    print('\t\tnumer of classes: %s' % classes)

    print('\tmodule ModelHandler.py: ')
    model = ModelHandler(classes=classes,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         metrics=metric,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         multi_label_lvl=multilabel_lvl)

    ### load parameters if already trained, otherwise train
    model = load_or_train_model(model=model,
                                dataset=dataset,
                                path=path,
                                param_folder_name=param_folder_name,
                                mode='per_lvl',
                                epochs=epochs,
                                ext_storage_path=ext_storage_path,
                                app='_%s' % taxa)

    # x = list(data_handler.samples_per_class.keys())
    # y = list(data_handler.samples_per_class.values())

    # ax = fig.add_subplot(2, 3, i+1)
    # title = 'Acc %s' %taxa
    # plot_classification_acc(x=x,
    #                 y=y,
    #                 colors='b',
    #                 title=title,
    #                 axis=ax)
    # for cl in data_handler.samples_per_class:
    #     print("not resampled %s --- %s: %d" % (taxa, cl, data_handler.samples_per_class[cl]))
    #     print("    resampled %s --- %s: %d" % (taxa, cl, data_handler.samples_per_class_normalized[cl]))

    val_names, val_accs = model.evaluate(model.net, data_handler.test_data, model.ctx, metric=metric)
    print('%s: %s' % (taxa, model.metric_str(val_names, val_accs)))

    print('------------------------------------------')
#
# plt.tight_layout()
# plt.show()

###################################################################################
#######################   All in One Classifier  ##################################
###################################################################################
# PARAMETERS Training
multilabel_lvl = 2

# epochs = 10
# # learning_rate = 0.001
# learning_rate = 0.1
# momentum =0.8
#
# param_folder_name = 'ParameterFiles_%s_e%i_lr%f_m%f'%(net_name,epochs,learning_rate,momentum)

num_workers = cpu_count()
num_gpus = 1
# batch_size = 2006
per_device_batch_size = 5
batch_size = per_device_batch_size * max(num_gpus, 1)

# PARAMETERS Model
# metric = mx.metric.Accuracy()
# metric = F1(average="micro")
# metric = F1()
metric = PCC()

# binary relevance approach -> ignores possible correlations

# PARAMETERS Augmentation
jitter_param = 0.4
lighting_param = 0.1

# fig = plt.figure(figsize=(15, 10))


print('working in all-in-one taxonomic classification')
print('\tmodule DataPrep.py: ... prepraring data')

data_prepper = DataPrep(rank=None, path=path, dataset=dataset, df=df, multilabel_lvl=multilabel_lvl,
                        taxonomic_groups=taxonomic_groups)

print('\tmodule DataHandler.py: ... loading image folder dataset and augmenting')
data_handler = DataHandler(path=data_prepper.imagefolder_path,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           augment=augment)
classes = data_handler.classes
print('\t\tnumer of classes: %s' % classes)

print('\tmodule ModelHandler.py: ')
model = ModelHandler(classes=classes,
                     batch_size=batch_size,
                     num_workers=num_workers,
                     metrics=metric,
                     learning_rate=learning_rate,
                     momentum=momentum,
                     multi_label_lvl=multilabel_lvl)

### load parameters if already trained, otherwise train
model = load_or_train_model(model=model,
                            dataset=dataset,
                            path=path,
                            param_folder_name=param_folder_name,
                            mode='all-in-one',
                            epochs=epochs,
                            ext_storage_path=ext_storage_path )

# x = list(data_handler.samples_per_class.keys())
# y = list(data_handler.samples_per_class.values())


# ax = fig.add_subplot(2, 3, i+1)
# title = 'Acc in ml.....strange yet'
# plot_classification_acc(x=x,
#                 y=y,
#                 colors='b',
#                 title=title,
#                 axis=ax)

for cl in data_handler.samples_per_class:
    print("not resampled --- %s: %d" % (cl, data_handler.samples_per_class[cl]))
    print("    resampled --- %s: %d" % (cl, data_handler.samples_per_class_normalized[cl]))

val_names, val_accs = model.evaluate(model.net, data_handler.test_data, model.ctx, metric=metric)
print(' %s' % (model.metric_str(val_names, val_accs)))

# plt.tight_layout()
# plt.show()
