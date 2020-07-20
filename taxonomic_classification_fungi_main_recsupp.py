import os
import pandas as pd
from multiprocessing import cpu_count
from matplotlib import pyplot as plt
import mxnet as mx
import numpy as np
from mxnet.metric import PCC
####
from mxnet import gluon
import skimage.io
####
## load custom modules
# from DataHandler import DataHandler
# from DataPrep import DataPrep
from DataRecHandler import DataRecHandler
from ModelHandler import ModelHandler
from collections import Counter


def load_or_train_model(model, dataset, mode, epochs, ext_storage_path, data_handler, fold=0, taxa=''):
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

def load_best_model(ext_storage_path, rank, mode, dataset, fold=0):
    param_file_name = '%s_%s%s.param' % (dataset, mode, rank)

    csv_file_name = param_file_name.split('.')[0] + '_f' + str(fold) + '.csv'
    df = pd.read_csv(os.path.join(ext_storage_path, csv_file_name))

    epoch = df['epochs'].loc[df['scores_test']==np.nanmax(df['scores_test'])]
    epoch = epoch.values[0]

    e_param_file_name = param_file_name.split('.')[0] + '_e' + str(epoch) + '_f' + str(fold) + '.param'
    abs_path_param_file_name = os.path.join(ext_storage_path, e_param_file_name)

    return abs_path_param_file_name

# Metaparameters for file handling
# path = "/home/stillsen/Documents/Data/Fungi_IC__new_set"
# path = '/home/stillsen/Documents/Data/rec'
path = '/home/stillsen/Documents/Data/rec'
# ext_storage_path = '/media/stillsen/Elements SE/Data/Fungi_IC__new_set/ParamData'
# ext_storage_path = '/media/stillsen/Elements SE/Data/Fungi_newset_recs/ParamData'
ext_storage_path = '/home/stillsen/Documents/Data/rec/ParamData'
# missing_values = ['', 'unknown', 'unclassified']
missing_values = ['', 'unknown', 'unclassified', 'unidentified']
taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
# augment = 'transform'
oversample_technique = 'transformOversampled' # 'naiveOversampled', 'smote'
dataset = 'fun'
net_name = 'densenet169'
# net_name = 'resnet152_v1d'

# csv_path = os.path.join(path, 'im.merged.v10032020_unique_id_set.csv')
csv_path = os.path.join(path, 'im.merged.v10032020_unique_id_set.csv')
df = pd.read_csv(csv_path, na_values=missing_values)

print('NaNs in the label data set')
print(df.iloc[:, 16:22].isnull().sum())
p = set(df['phylum'].values)
print('phylum: %i :: %s' %(len(p),p))
c = set(df['class'].values)
print('class: %i :: %s' %(len(c),c))
o = set(df['order'].values)
print('order: %i :: %s' %(len(o),o))
f = set(df['family'].values)
print('family: %i :: %s' %(len(f),f))
g = set(df['genus'].values)
print('genus: %i :: %s' %(len(g),g))
s = set(df['species'].values)
print('species: %i :: %s' %(len(s),s))
classes = [len(p), len(c), len(o), len(f), len(g), len(s)]

###################################################################################
#######################   Per Level Classifier  ###################################
###################################################################################
# PARAMETERS Training
# multilabel lvl = 1 #==> per rank classifier
# multilabel_lvl = 2 # ==> all-in-one classifier
multilabel_lvl = 1

# k fold cross validation
k = 5

epochs = 20
# learning_rate = 0.001
learning_rate = 0.0001
# learning_rate = 0.4
momentum = 0.8

param_folder_name = 'ParameterFiles_%s_e%i_lr%f_m%f' % (net_name, epochs, learning_rate, momentum)

num_workers = cpu_count()
num_gpus = 1
# batch_size = 2006
per_device_batch_size = 5
batch_size = per_device_batch_size * max(num_gpus, 1)
# batch_size = 1
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

#
#
# ###################################################################################
# print("########### X Val ###############")
# ###################################################################################
# oversample_technique = 'naiveOversampled'
#
# for rank_idx, taxa in enumerate(taxonomic_groups):
#     # print('\tmodule DataPrep.py: ... prepraring data')
#     # data_prepper = DataPrep(rank=taxa, path=path, dataset=dataset, df=df, multilabel_lvl=multilabel_lvl,
#     #                         taxonomic_groups=taxonomic_groups)
#     #
#     # print('\tmodule DataHandler.py: ... loading image folder dataset and augmenting')
#     # data_handler = DataHandler(path=data_prepper.imagefolder_path,
#     #                            batch_size=batch_size,
#     #                            num_workers=num_workers,
#     #                            augment=augment)
#     # classes = data_handler.classes
#     # if taxa == 'phylum' or taxa == 'class' or taxa == 'order' or taxa == 'family' or taxa == 'genus':
#     #     print('skipping %s'%taxa)
#     #     continue
#     # if taxa == 'species':
#     #     s = 3
#     # else:
#     s = 0
#     print('%i-fold crossvalidation for %s' % (k, taxa))
#     for fold in range(s,k):
#         if fold == 0:
#             create_recs = True
#         else:
#             create_recs = False
#         print('fold %s'%fold)
#         data_rec_handler = DataRecHandler(root_path=path,
#                                           rank_name=taxa,  # set to 'all-in-one', for multilabel_lvl=2
#                                           rank_idx=rank_idx,
#                                           batch_size=batch_size,
#                                           num_workers=num_workers,
#                                           k=k,
#                                           create_recs=create_recs,
#                                           oversample_technique=oversample_technique)
#         data_rec_handler.load_rec(fold)
#
#         print('number of classes %i' % classes[rank_idx])
#         print('\tmodule ModelHandler.py: ')
#         model = ModelHandler(classes=classes[rank_idx],
#                              batch_size=batch_size,
#                              num_workers=num_workers,
#                              metrics=metric,
#                              learning_rate=learning_rate,
#                              momentum=momentum,
#                              multi_label_lvl=multilabel_lvl,
#                              model_name=net_name,
#                              rank_idx=rank_idx,
#                              wd=0.01)
#         ### load parameters if already trained, otherwise train
#         model = load_or_train_model(model=model,
#                                     dataset=dataset,
#                                     # mode='per_lvl',
#                                     mode='per_lvl_xval_oversample',
#                                     epochs=epochs,
#                                     ext_storage_path=ext_storage_path,
#                                     taxa='_%s' % taxa,
#                                     data_handler=data_rec_handler,
#                                     fold=fold)
#
#


# ###################################################################################
# print("############ tt-split #############")
# ###################################################################################
# print("Per Level Classifier  ")
# for rank_idx, taxa in enumerate(taxonomic_groups):
#     print('t-split for %s' % (taxa))
#     # taxa = 'genus'
#     # rank_idx = 4
#     create_recs = True
#     data_rec_handler = DataRecHandler(root_path=path,
#                                       rank_name=taxa,  # set to 'all-in-one', for multilabel_lvl=2
#                                       rank_idx=rank_idx,
#                                       batch_size=batch_size,
#                                       num_workers=num_workers,
#                                       k=k,
#                                       create_recs=create_recs,
#                                       oversample_technique=oversample_technique)
#     data_rec_handler.load_rec()
#     print('number of classes %i' %classes[rank_idx])
#
#     print('\tmodule ModelHandler.py: ')
#     model = ModelHandler(classes=classes[rank_idx],
#                          batch_size=batch_size,
#                          num_workers=num_workers,
#                          metrics=metric,
#                          learning_rate=learning_rate,
#                          momentum=momentum,
#                          multi_label_lvl=multilabel_lvl,
#                          model_name=net_name,
#                          rank_idx=rank_idx,
#                          wd=0.0001)
#
#     ### load parameters if already trained, otherwise train
#     model = load_or_train_model(model=model,
#                                 dataset=dataset,
#                                 mode='per_lvl_tt-split',
#                                 epochs=epochs,
#                                 ext_storage_path=ext_storage_path,
#                                 taxa='_%s' % taxa,
#                                 data_handler=data_rec_handler)

print("########### Nested ###############")
multilabel_lvl = 3 # relevant for model
taxa = 'hierarchical' #relevant for RecordIO creation
prev_best_model = None
oversample_technique='naiveOversampled'
# oversample_technique=None
# k=7
# --> set dataRecHandler to appropriate --------------------> Each Level Needs it's own dataset: simulate with crossval and fold, k=7 <---------------
# also for orig use mode ='_oversampled_xval_ML' but with oversample commented out
for rank_idx, taxa_rank in enumerate(taxonomic_groups):
    print('hierarchical tt-split for %s %s' %(taxa, taxa_rank))

    if rank_idx==0:
        create_recs = True
        data_rec_handler = DataRecHandler(root_path=path,
                                          rank_name=taxa,  # set to 'all-in-one', for multilabel_lvl=2
                                          rank_idx=rank_idx,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          k=k,
                                          create_recs=create_recs,
                                          oversample_technique=oversample_technique)
    else:
        create_recs = False

    data_rec_handler.load_rec(rank_idx)

    print('number of classes %i' %data_rec_handler.classes[rank_idx])
    if rank_idx == 0:
        model = ModelHandler(classes=classes[rank_idx],
                             batch_size=batch_size,
                             num_workers=num_workers,
                             metrics=metric,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             multi_label_lvl=multilabel_lvl,
                             model_name=net_name,
                             rank_idx=rank_idx,
                             prior_param=prev_best_model,
                             wd=0.01)
    else:
        model.add_layer(prior_param=prev_best_model,
                        rank_idx=rank_idx,
                        classes=classes[rank_idx])

    if taxa_rank == 'phylum' or taxa_rank == 'class':
        prev_best_model = load_best_model(ext_storage_path=ext_storage_path,
                                          rank='_%s' % taxa_rank,
                                          dataset=dataset,
                                          mode='chained_per-lvl')
        model.net.load_parameters(prev_best_model)
    else:
        model = load_or_train_model(model=model,
                                    dataset=dataset,
                                    mode='chained_per-lvl',
                                    epochs=epochs,
                                    ext_storage_path=ext_storage_path,
                                    taxa='_%s' % taxa_rank,
                                    data_handler=data_rec_handler)
    if model.best_model is not None:
        prev_best_model = os.path.join(ext_storage_path, model.best_model)
        print('previous best model: %s'%prev_best_model)

# ##################################################################################
# ######################   All in One Classifier  ##################################
# ##################################################################################
# # PARAMETERS Training
# multilabel_lvl = 2
# oversample_technique = 'naiveOversampled'
# taxa = 'all-in-one'
# rank_idx=0
# fold=0
# print('working in all-in-one taxonomic classification')
# print('%i-fold crossvalidation for %s' % (k, taxa))
# # for fold in range(k):
# #     if fold == 0:
# #         create_recs = True
# #     else:
# #         create_recs = False
# create_recs = True
# data_rec_handler = DataRecHandler(root_path=path,
#                                   rank_name=taxa,  # set to 'all-in-one', for multilabel_lvl=2
#                                   rank_idx=rank_idx,
#                                   batch_size=batch_size,
#                                   num_workers=num_workers,
#                                   k=k,
#                                   create_recs=create_recs,
#                                   oversample_technique=oversample_technique)
# data_rec_handler.load_rec(fold)
# print('number of classes %i' %classes[rank_idx])
# model = ModelHandler(classes=classes[rank_idx],
#                      batch_size=batch_size,
#                      num_workers=num_workers,
#                      metrics=metric,
#                      learning_rate=learning_rate,
#                      momentum=momentum,
#                      multi_label_lvl=multilabel_lvl,
#                      model_name=net_name,
#                      rank_idx=rank_idx,
#                      wd=0.01)
#
# model = load_or_train_model(model=model,
#                             dataset=dataset,
#                             mode='all-in-one',
#                             epochs=epochs,
#                             ext_storage_path=ext_storage_path,
#                             taxa='_%s' % taxa,
#                             data_handler=data_rec_handler,
#                             fold=fold)

