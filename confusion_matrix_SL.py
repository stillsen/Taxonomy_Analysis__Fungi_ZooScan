from sklearn.metrics import confusion_matrix
# from pandas_ml import ConfusionMatrix
import numpy as np
import mxnet as mx
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import init, gluon
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms
import os
from mxnet.io import ImageRecordIter
from collections import Counter
def plot_confusion_matrix(y_true,
                          y_pred,
                          target_names,
                          path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a true and precition vector, compute confusion matrix and make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    from sklearn.metrics import matthews_corrcoef

    cm = confusion_matrix(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\nMCC/PCC={:0.4f}'.format(accuracy, misclass, mcc))
    plt.savefig(os.path.join(path, title+'.png'), bbox_inches='tight')
    plt.show()



path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/p17'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/p17/test'
rec_prefix = 'phylum_test_oversampled_tt-split_SL'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_per_lvl_tt-split_phylum_e17_f0.param'

label_offset = [0, 6, 19, 50, 106, 194]


############# load net ###############
classes = 6
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
finetune_net = get_model('densenet169', classes=classes)
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
finetune_net.output.collect_params().setattr('lr_mult', 10)
finetune_net.features = pretrained_net.features
# finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()
finetune_net.load_parameters(os.path.join(path,param_file))

############# load data ##############
test_data = ImageRecordIter(
    path_imgrec=os.path.join(rec_path, rec_prefix+'.rec'),
    path_imgidx=os.path.join(rec_path, rec_prefix+'.idx'),
    path_imglist=os.path.join(rec_path, rec_prefix+'.lst'),
    aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
    data_shape=(3, 224, 224),
    batch_size=1,
    resize=224,
    label_width=1,
    shuffle=False  # train true, test no
)

phylum_labels = []
preds = []

for i,batch in enumerate(test_data):
    phylum_label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

    phylum_labels.append(phylum_label)
    pred = finetune_net(image[0])
    preds.append(pred[0].asnumpy().argmax()+label_offset[0])

cm_path = os.path.join(path,'cms')
if not os.path.exists(cm_path):
    os.mkdir(cm_path)

target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(phylum_labels)))]
plot_confusion_matrix(
    y_true=phylum_labels,
    y_pred=preds,
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Phylum',
    normalize=False)

# path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/c14'
# rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/c14/test'
# rec_prefix = 'class_test_oversampled_tt-split_SL'
# global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
# param_file = 'fun_per_lvl_tt-split_class_e14_f0.param'
#
#
# ############# load net ###############
# classes = 13
# gpus = mx.test_utils.list_gpus()
# ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
# pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
# finetune_net = get_model('densenet169', classes=classes)
# finetune_net.output.initialize(init.Xavier(), ctx=ctx)
# finetune_net.output.collect_params().setattr('lr_mult', 10)
# finetune_net.features = pretrained_net.features
# # finetune_net.collect_params().reset_ctx(ctx)
# finetune_net.hybridize()
# finetune_net.load_parameters(os.path.join(path,param_file))
#
# ############# load data ##############
# test_data = ImageRecordIter(
#     path_imgrec=os.path.join(rec_path, rec_prefix+'.rec'),
#     path_imgidx=os.path.join(rec_path, rec_prefix+'.idx'),
#     path_imglist=os.path.join(rec_path, rec_prefix+'.lst'),
#     aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
#     data_shape=(3, 224, 224),
#     batch_size=1,
#     resize=224,
#     label_width=1,
#     shuffle=False  # train true, test no
# )
#
# class_labels = []
# preds = []
#
# for i,batch in enumerate(test_data):
#     class_label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
#     image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
#
#     class_labels.append(class_label)
#     pred = finetune_net(image[0])
#     preds.append(pred[0].asnumpy().argmax()+label_offset[1])
#     # preds.append(pred[0].asnumpy().argmax())
#     print('l: %s, p: %s'%(len(class_labels), len(preds)))
#     if len(class_labels) != len(preds):
#         print('issoweit')
#     if i == 179:
#         print('issoweit')
#
#
# cm_path = os.path.join(path,'cms')
# if not os.path.exists(cm_path):
#     os.mkdir(cm_path)
#
# labels = class_labels[:]
# if not all(p in class_labels for p in preds):
#     # yet_to_add = [p not in class_labels for p in preds]
#     labels.append(preds[[p not in labels for p in preds].index(True)])
# target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(labels)))]
# print('target names %s'%(len(target_names)))
# plot_confusion_matrix(
#     y_true=class_labels,
#     y_pred=preds,
#     # target_names=list(phylum_taxon),
#     target_names=target_names,
#     path=cm_path,
#     title='Confusion_Matrix-Class',
#     normalize=False)
#
# path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/o11'
# rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/o11/test'
# rec_prefix = 'order_test_oversampled_tt-split_SL'
# global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
# param_file = 'fun_per_lvl_tt-split_order_e11_f0.param'
#
#
# ############# load net ###############
# classes = 31
# gpus = mx.test_utils.list_gpus()
# ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
# pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
# finetune_net = get_model('densenet169', classes=classes)
# finetune_net.output.initialize(init.Xavier(), ctx=ctx)
# finetune_net.output.collect_params().setattr('lr_mult', 10)
# finetune_net.features = pretrained_net.features
# # finetune_net.collect_params().reset_ctx(ctx)
# finetune_net.hybridize()
# finetune_net.load_parameters(os.path.join(path,param_file))
#
# ############# load data ##############
# test_data = ImageRecordIter(
#     path_imgrec=os.path.join(rec_path, rec_prefix+'.rec'),
#     path_imgidx=os.path.join(rec_path, rec_prefix+'.idx'),
#     path_imglist=os.path.join(rec_path, rec_prefix+'.lst'),
#     aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
#     data_shape=(3, 224, 224),
#     batch_size=1,
#     resize=224,
#     label_width=1,
#     shuffle=False  # train true, test no
# )
#
# order_labels = []
# preds = []
#
# for i,batch in enumerate(test_data):
#     order_label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
#     image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
#
#     order_labels.append(order_label)
#     pred = finetune_net(image[0])
#     preds.append(pred[0].asnumpy().argmax()+label_offset[2])
#     # preds.append(pred[0].asnumpy())
#     print('l: %s, p: %s'%(len(order_labels), len(preds)))
# cm_path = os.path.join(path,'cms')
# if not os.path.exists(cm_path):
#     os.mkdir(cm_path)
#
# labels = order_labels[:]
# if not all(p in labels for p in preds):
#     # yet_to_add = [p not in class_labels for p in preds]
#     while not all(p in labels for p in preds):
#         labels.append(preds[[p not in labels for p in preds].index(True)])
# target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(labels)))]
# plot_confusion_matrix(
#     y_true=order_labels,
#     y_pred=preds,
#     # target_names=list(phylum_taxon),
#     target_names=target_names,
#     path=cm_path,
#     title='Confusion_Matrix-Order',
#     normalize=False)

