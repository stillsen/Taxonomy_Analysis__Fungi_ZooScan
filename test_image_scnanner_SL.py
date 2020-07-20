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


for i,batch in enumerate(test_data):
    phylum_label_id = gluon.utils.split_and_load(batch.label[0][:, 0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    phylum_label = global_mapping.loc[global_mapping['id'] == phylum_label_id]['taxon'].values[0]

    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
    img = np.moveaxis(image[0][0].asnumpy(),0,2).astype(float)/255

    p_preds = finetune_net(image[0])
    phylum_prediction_id = p_preds[0].asnumpy().argmax()

    phylum_prediction_label = global_mapping.loc[global_mapping['id']==phylum_prediction_id]['taxon'].values[0]

    score = [phylum_label == phylum_prediction_label,
             class_label == class_prediction_label,
             order_label == order_prediction_label,
             family_label == family_prediction_label,
             genus_label == genus_prediction_label,
             species_label == species_prediction_label]
    score = sum(score)/6

    plt.imshow(img)

    text_str = str(i) + ')  score:  ' + str(score)[:4] +'\n'\
               'taxon: label - prediction\n'+\
               'p:  ' + phylum_label + ' - ' + phylum_prediction_label +'\n'+\
               'c:  ' + class_label + ' - ' + class_prediction_label + '\n'+\
               'o:  ' + order_label + ' - ' + order_prediction_label + '\n'+\
               'f:  ' + family_label + ' - ' + family_prediction_label + '\n'+\
               'g:  ' + genus_label + ' - ' + genus_prediction_label + '\n'+\
               's:  ' + species_label + ' - ' + species_prediction_label
    plt.text(2, 80, text_str, color='white', )
    # plt.show()
    tis_path = os.path.join(path,'test_image_scan')
    if not os.path.exists(tis_path):
        os.mkdir(tis_path)
    best_path = os.path.join(tis_path,'best_score')
    if not os.path.exists(best_path):
        os.mkdir(best_path)
    fn = str(i) + '_' + phylum_label + '_' + class_label + '_' + order_label + '_' + family_label + '_' + genus_label + '_' + species_label + '.png'
    print('saving ' + fn)
    if score >= 0.8:
        best_scores_id.append(i)
        plt.savefig(os.path.join(best_path, fn))
    else:
        plt.savefig(os.path.join(tis_path,fn))
    plt.clf()
print('best scores id: %s' % best_scores_id)
#########################################################################################################
path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/c14'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/c14/test'
rec_prefix = 'class_test_oversampled_tt-split_SL'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_per_lvl_tt-split_class_e14_f0.param'


############# load net ###############
classes = 13
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

class_labels = []
preds = []

for i,batch in enumerate(test_data):
    class_label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

    class_labels.append(class_label)
    pred = finetune_net(image[0])
    preds.append(pred[0].asnumpy().argmax()+label_offset[1])
    # preds.append(pred[0].asnumpy().argmax())
    print('l: %s, p: %s'%(len(class_labels), len(preds)))
    if len(class_labels) != len(preds):
        print('issoweit')
    if i == 179:
        print('issoweit')


cm_path = os.path.join(path,'cms')
if not os.path.exists(cm_path):
    os.mkdir(cm_path)

labels = class_labels[:]
if not all(p in class_labels for p in preds):
    # yet_to_add = [p not in class_labels for p in preds]
    labels.append(preds[[p not in labels for p in preds].index(True)])
target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(labels)))]
print('target names %s'%(len(target_names)))
plot_confusion_matrix(
    y_true=class_labels,
    y_pred=preds,
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Class',
    normalize=False)

path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/o11'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/o11/test'
rec_prefix = 'order_test_oversampled_tt-split_SL'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_per_lvl_tt-split_order_e11_f0.param'


############# load net ###############
classes = 31
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

order_labels = []
preds = []

for i,batch in enumerate(test_data):
    order_label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

    order_labels.append(order_label)
    pred = finetune_net(image[0])
    preds.append(pred[0].asnumpy().argmax()+label_offset[2])
    # preds.append(pred[0].asnumpy())
    print('l: %s, p: %s'%(len(order_labels), len(preds)))
cm_path = os.path.join(path,'cms')
if not os.path.exists(cm_path):
    os.mkdir(cm_path)

labels = order_labels[:]
if not all(p in labels for p in preds):
    # yet_to_add = [p not in class_labels for p in preds]
    while not all(p in labels for p in preds):
        labels.append(preds[[p not in labels for p in preds].index(True)])
target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(labels)))]
plot_confusion_matrix(
    y_true=order_labels,
    y_pred=preds,
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Order',
    normalize=False)

