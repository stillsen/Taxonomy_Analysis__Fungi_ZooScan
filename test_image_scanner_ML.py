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



path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/ML_e4_naiveOversampled_tt-split_ML_lr001_wd01'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/ML_e4_naiveOversampled_tt-split_ML_lr001_wd01/test'
rec_prefix = 'all-in-one_test_oversampled_tt-split_ML'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_all-in-one_all-in-one_e4_f0.param'

label_offset = [0, 6, 19, 50, 106, 194]

############# load net ###############
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
finetune_net = BigBangNet(p=6,
                          c=13,
                          o=31,
                          f=56,
                          g=88,
                          s=166)
finetune_net.collect_params().initialize(init.Xavier(), ctx=ctx)
finetune_net.feature = pretrained_net.features
finetune_net.hybridize()
finetune_net.load_parameters(os.path.join(path, param_file))

############# load data ##############
test_data = ImageRecordIter(
    path_imgrec=os.path.join(rec_path, rec_prefix+'.rec'),
    path_imgidx=os.path.join(rec_path, rec_prefix+'.idx'),
    path_imglist=os.path.join(rec_path, rec_prefix+'.lst'),
    aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
    data_shape=(3, 224, 224),
    batch_size=1,
    resize=224,
    label_width=6,
    shuffle=False  # train true, test no
)


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

    preds = finetune_net(image[0])

    phylum_prediction_id = preds[0][0].asnumpy().argmax()
    class_prediction_id = preds[1][0].asnumpy().argmax() + label_offset[1]
    order_prediction_id = preds[2][0].asnumpy().argmax() + label_offset[2]
    family_prediction_id = preds[3][0].asnumpy().argmax() + label_offset[3]
    genus_prediction_id = preds[4][0].asnumpy().argmax() + label_offset[4]
    species_prediction_id = preds[5][0].asnumpy().argmax() + label_offset[5]

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
    # text_str = str(i) + ') taxon: label - prediction\n'+\
    #            'p:  ' + phylum_label + ' - ' + phylum_prediction_label + ' ' + str(preds[0][0].asnumpy()[phylum_prediction_id])[:4] +'\n'+\
    #            'c:  ' + class_label + ' - ' + class_prediction_label + ' ' + str(preds[1][0].asnumpy()[class_prediction_id-label_offset[1]])[:4] + '\n'+\
    #            'o:  ' + order_label + ' - ' + order_prediction_label + ' ' + str(preds[2][0].asnumpy()[order_prediction_id-label_offset[2]])[:4] + '\n'+\
    #            'f:  ' + family_label + ' - ' + family_prediction_label + ' ' + str(preds[3][0].asnumpy()[family_prediction_id-label_offset[3]])[:4] + '\n'+\
    #            'g:  ' + genus_label + ' - ' + genus_prediction_label + ' ' + str(preds[4][0].asnumpy()[genus_prediction_id-label_offset[4]])[:4] + '\n'+\
    #            's:  ' + species_label + ' - ' + species_prediction_label +' ' + str(preds[5][0].asnumpy()[species_prediction_id-label_offset[5]])[:4]
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
    fn = str(i) + '_' + phylum_label + '_' + class_label + '_' + order_label + '_' + family_label + '_' + genus_label + '_' + species_label + '.png'
    print('saving ' + fn)
    plt.savefig(os.path.join(tis_path,fn))
    plt.clf()