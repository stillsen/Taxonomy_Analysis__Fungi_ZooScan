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
from ModelHandler import BigBangNet
from mxnet import init, nd, ndarray, gluon

path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/HC_naiveOversampled_tt-split_HC_lr001_wd01'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/HC_naiveOversampled_tt-split_HC_lr001_wd01/hierarchical'
p_rec_prefix = 'fun_chained_per-lvl_phylum_e11_f0'
# p_rec_prefix = 'fun_chained_per-lvl_phylum_e18_f0'
c_rec_prefix = 'fun_chained_per-lvl_class_e14_f0'
o_rec_prefix = 'fun_chained_per-lvl_order_e18_f0'
f_rec_prefix = 'fun_chained_per-lvl_family_e10_f0'
g_rec_prefix = 'fun_chained_per-lvl_genus_e5_f0'
s_rec_prefix = 'fun_chained_per-lvl_species_e5_f0'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))

label_offset = [0, 6, 19, 50, 106, 194]

############# load net ###############
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]

pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
p_finetune_net = get_model('densenet169', classes=6)
p_finetune_net.collect_params().initialize(init.Xavier(), ctx=ctx)
p_finetune_net.features = pretrained_net.features
p_finetune_net.hybridize()
p_finetune_net.load_parameters(os.path.join(path, p_rec_prefix+'.param'))

c_finetune_net = gluon.nn.HybridSequential()
with c_finetune_net.name_scope():
    c_finetune_net.add(p_finetune_net)
    next_layer = gluon.nn.Dense(13)
    c_finetune_net.add(next_layer)
c_finetune_net.collect_params().initialize()
c_finetune_net.hybridize()
c_finetune_net.load_parameters(os.path.join(path, c_rec_prefix+'.param'))

o_finetune_net = gluon.nn.HybridSequential()
with o_finetune_net.name_scope():
    o_finetune_net.add(c_finetune_net)
    next_layer = gluon.nn.Dense(31)
    o_finetune_net.add(next_layer)
o_finetune_net.collect_params().initialize()
o_finetune_net.hybridize()
o_finetune_net.load_parameters(os.path.join(path, o_rec_prefix+'.param'))

f_finetune_net = gluon.nn.HybridSequential()
with f_finetune_net.name_scope():
    f_finetune_net.add(o_finetune_net)
    next_layer = gluon.nn.Dense(56)
    f_finetune_net.add(next_layer)
f_finetune_net.collect_params().initialize()
f_finetune_net.hybridize()
f_finetune_net.load_parameters(os.path.join(path, f_rec_prefix+'.param'))

g_finetune_net = gluon.nn.HybridSequential()
with g_finetune_net.name_scope():
    g_finetune_net.add(f_finetune_net)
    next_layer = gluon.nn.Dense(88)
    g_finetune_net.add(next_layer)
g_finetune_net.collect_params().initialize()
g_finetune_net.hybridize()
g_finetune_net.load_parameters(os.path.join(path, g_rec_prefix+'.param'))

s_finetune_net = gluon.nn.HybridSequential()
with s_finetune_net.name_scope():
    s_finetune_net.add(g_finetune_net)
    next_layer = gluon.nn.Dense(166)
    s_finetune_net.add(next_layer)
s_finetune_net.collect_params().initialize()
s_finetune_net.hybridize()
s_finetune_net.load_parameters(os.path.join(path, s_rec_prefix+'.param'))

############# load data ##############
test_data = ImageRecordIter(
    path_imgrec=os.path.join(rec_path, 'hierarchical_orig_tt-split_ML_test.rec'),
    path_imgidx=os.path.join(rec_path, 'hierarchical_orig_tt-split_ML_test.idx'),
    path_imglist=os.path.join(rec_path, 'hierarchical_orig_tt-split_ML_test.lst'),
    aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
    data_shape=(3, 224, 224),
    batch_size=1,
    resize=224,
    label_width=6,
    shuffle=False  # train true, test no
)

best_scores_id = []
for i,batch in enumerate(test_data):
    phylum_label_id = gluon.utils.split_and_load(batch.label[0][:, 0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    class_label_id = gluon.utils.split_and_load(batch.label[0][:, 1], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    order_label_id = gluon.utils.split_and_load(batch.label[0][:, 2], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    family_label_id = gluon.utils.split_and_load(batch.label[0][:, 3], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    genus_label_id = gluon.utils.split_and_load(batch.label[0][:, 4], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    species_label_id = gluon.utils.split_and_load(batch.label[0][:, 5], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()

    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
    img = np.moveaxis(image[0][0].asnumpy(),0,2).astype(float)/255

    phylum_label = global_mapping.loc[global_mapping['id']==phylum_label_id]['taxon'].values[0]
    class_label = global_mapping.loc[global_mapping['id']==class_label_id]['taxon'].values[0]
    order_label = global_mapping.loc[global_mapping['id']==order_label_id]['taxon'].values[0]
    family_label = global_mapping.loc[global_mapping['id']==family_label_id]['taxon'].values[0]
    genus_label = global_mapping.loc[global_mapping['id']==genus_label_id]['taxon'].values[0]
    species_label = global_mapping.loc[global_mapping['id']==species_label_id]['taxon'].values[0]

    p_preds = p_finetune_net(image[0])
    c_preds = c_finetune_net(image[0])
    o_preds = o_finetune_net(image[0])
    f_preds = f_finetune_net(image[0])
    g_preds = g_finetune_net(image[0])
    s_preds = s_finetune_net(image[0])

    phylum_prediction_id = p_preds[0].asnumpy().argmax()
    class_prediction_id = c_preds[0].asnumpy().argmax() + label_offset[1]
    order_prediction_id = o_preds[0].asnumpy().argmax() + label_offset[2]
    family_prediction_id = f_preds[0].asnumpy().argmax() + label_offset[3]
    genus_prediction_id = g_preds[0].asnumpy().argmax() + label_offset[4]
    species_prediction_id = s_preds[0].asnumpy().argmax() + label_offset[5]

    phylum_prediction_label = global_mapping.loc[global_mapping['id']==phylum_prediction_id]['taxon'].values[0]
    class_prediction_label = global_mapping.loc[global_mapping['id']==class_prediction_id]['taxon'].values[0]
    order_prediction_label = global_mapping.loc[global_mapping['id']==order_prediction_id]['taxon'].values[0]
    family_prediction_label = global_mapping.loc[global_mapping['id']==family_prediction_id]['taxon'].values[0]
    genus_prediction_label = global_mapping.loc[global_mapping['id']==genus_prediction_id]['taxon'].values[0]
    species_prediction_label = global_mapping.loc[global_mapping['id']==species_prediction_id]['taxon'].values[0]

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