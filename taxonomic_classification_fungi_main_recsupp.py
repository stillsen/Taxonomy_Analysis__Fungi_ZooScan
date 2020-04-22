import os
import pandas as pd
from multiprocessing import cpu_count
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet.metric import PCC

# load custom modules
# from DataHandler import DataHandler
# from DataPrep import DataPrep
from DataRecHandler import DataRecHandler
from ModelHandler import ModelHandler


def load_or_train_model(model, dataset, mode, epochs, ext_storage_path, data_handler, fold, taxa=''):
    param_file_name = '%s_%s%s.param' % (dataset, mode, taxa)
    abs_param_file_name = os.path.join(ext_storage_path, param_file_name)
    print('Ext Storage Path: %s'%ext_storage_path)
    if os.path.exists(abs_param_file_name):
        print('\tloading %s' % param_file_name)
        model.net.load_parameters(abs_param_file_name)
    else:  # train
        print('\ttraining %s' % param_file_name)
        model.net = model.train(train_iter=data_handler.train_data,
                                val_iter=data_handler.test_data,
                                epochs=epochs,
                                param_file_name=param_file_name,
                                ext_storage_path=ext_storage_path,
                                fold=fold)
    return model


# Metaparameters for file handling
# path = "/home/stillsen/Documents/Data/Fungi_IC__new_set"
path = '/home/stillsen/Documents/Data/rec'
# ext_storage_path = '/media/stillsen/Elements SE/Data/Fungi_IC__new_set/ParamData'
ext_storage_path = '/media/stillsen/Elements SE/Data/Fungi_newset_recs/ParamData'
# missing_values = ['', 'unknown', 'unclassified']
missing_values = ['', 'unknown', 'unclassified', 'unidentified']
taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
# augment = 'transform'
oversample_technique = 'transformed' # naive, smote
dataset = 'fun'
net_name = 'densenet169'

csv_path = os.path.join(path, 'im.merged.v10032020_unique_id_set.csv')
df = pd.read_csv(csv_path, na_values=missing_values)

print('NaNs in the label data set')
print(df.isnull().sum())

###################################################################################
#######################   Per Level Classifier  ###################################
###################################################################################
# PARAMETERS Training
# multilabel lvl = 1 ==> per rank classifier
# multilabel lvl = 2 ==> all-in-one classifier
multilabel_lvl = 1

# k fold cross validation
k = 5

epochs = 20
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
save_all = True

# PARAMETERS Model
# metric = mx.metric.Accuracy()
# metric = F1(average="micro")
# metric = F1()
metric = PCC()

# binary relevance approach -> ignores possible correlations

# PARAMETERS Augmentation
jitter_param = 0.4
lighting_param = 0.1

###########################################################################################################

k=5

print("########################################")
print("Per Level Classifier  ")
# fig = plt.figure(figsize=(15, 10))
# subplot = 1

for i, taxa in enumerate(taxonomic_groups):
    # print('\tmodule DataPrep.py: ... prepraring data')
    # data_prepper = DataPrep(rank=taxa, path=path, dataset=dataset, df=df, multilabel_lvl=multilabel_lvl,
    #                         taxonomic_groups=taxonomic_groups)
    #
    # print('\tmodule DataHandler.py: ... loading image folder dataset and augmenting')
    # data_handler = DataHandler(path=data_prepper.imagefolder_path,
    #                            batch_size=batch_size,
    #                            num_workers=num_workers,
    #                            augment=augment)
    # classes = data_handler.classes
    print('%i-fold crossvalidation for %s' % (k, taxa))
    for fold in range(k):
        if fold == 0:
            data_rec_handler = DataRecHandler(root_path=path,
                                              rank=taxa,  # set to 'all-in-one', for multilabel_lvl=2
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              k=k,
                                              create_recs=True,
                                              oversample_technique=oversample_technique)
        data_rec_handler.load_rec(fold)
        print('\t\tnumer of classes: %s' % data_rec_handler.classes)

        print('\tmodule ModelHandler.py: ')
        model = ModelHandler(classes=data_rec_handler.classes,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             metrics=metric,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             multi_label_lvl=multilabel_lvl,
                             model_name=net_name)

        ### load parameters if already trained, otherwise train
        model = load_or_train_model(model=model,
                                    dataset=dataset,
                                    mode='per_lvl',
                                    epochs=epochs,
                                    ext_storage_path=ext_storage_path,
                                    taxa='_%s' % taxa,
                                    data_handler=data_rec_handler,
                                    fold=fold)


# ###################################################################################
# #######################   All in One Classifier  ##################################
# ###################################################################################
# PARAMETERS Training
multilabel_lvl = 2
taxa = 'all-in-one'

print('working in all-in-one taxonomic classification')
print('%i-fold crossvalidation for %s' % (k, taxa))
for fold in range(k):
    if fold == 0:
        data_rec_handler = DataRecHandler(root_path=path,
                                          rank=taxa,  # set to 'all-in-one', for multilabel_lvl=2
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          k=k,
                                          create_recs=True,
                                          oversample_technique=oversample_technique)
    data_rec_handler.load_rec(fold)
    print('\t\tnumer of classes: %s' % data_rec_handler.classes)

    print('\tmodule ModelHandler.py: ')
    model = ModelHandler(classes=data_rec_handler.classes,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         metrics=metric,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         multi_label_lvl=multilabel_lvl,
                         model_name=net_name)

    ### load parameters if already trained, otherwise train
    model = load_or_train_model(model=model,
                                dataset=dataset,
                                mode='all-in-one',
                                epochs=epochs,
                                ext_storage_path=ext_storage_path,
                                taxa='_%s' % taxa,
                                data_handler=data_rec_handler,
                                fold=fold)