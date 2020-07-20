from sklearn.metrics import confusion_matrix
# from pandas_ml import ConfusionMatrix
import numpy as np
import mxnet as mx
import pandas as pd
from matplotlib import pyplot as plt
from mxnet import init, gluon, nd
from gluoncv.model_zoo import get_model
from mxnet.gluon.data.vision import transforms
import os
from mxnet.io import ImageRecordIter
from collections import Counter
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import skimage.io, skimage.segmentation


def classifier_p(images):
    preds = p_finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    return preds.asnumpy()
def classifier_c(images):
    preds = c_finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    return preds.asnumpy()
def classifier_o(images):
    preds = o_finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    return preds.asnumpy()
def classifier_f(images):
    preds = f_finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    return preds.asnumpy()
def classifier_g(images):
    preds = g_finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    return preds.asnumpy()
def classifier_s(images):
    preds = s_finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    return preds.asnumpy()

kernel_size=6
max_dist=50
ratio=.4
neighborhood_size = 1000
superpixel_count = 100


taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
taxonomic_groups_to_color = {'phylum': 0.857142857142857, 'class': 0.714285714285714, 'order': 0.571428571428571,
                             'family': 0.428571428571429, 'genus': 0.285714285714286, 'species': 0.142857142857143}
label_offset = [0, 6, 19, 50, 106, 194]

cc = [taxonomic_groups_to_color[x] for x in taxonomic_groups]
cmap = plt.cm.get_cmap(name='Dark2', lut=6)
colors = cmap(cc)
colors = colors[:,:3]

###################################################
#################### PHYLUM #######################
###################################################
path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/p17'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/p17/test'
rec_prefix = 'phylum_test_oversampled_tt-split_SL'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_per_lvl_tt-split_phylum_e17_f0.param'

############# load net ###############
classes = 6
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
p_finetune_net = get_model('densenet169', classes=classes)
p_finetune_net.output.initialize(init.Xavier(), ctx=ctx)
p_finetune_net.output.collect_params().setattr('lr_mult', 10)
p_finetune_net.features = pretrained_net.features
# finetune_net.collect_params().reset_ctx(ctx)
p_finetune_net.hybridize()
p_finetune_net.load_parameters(os.path.join(path,param_file))
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
# errors = 0
# for i,batch in enumerate(test_data):
#         label= gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
#         image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
#
#         taxon = str(global_mapping.loc[global_mapping.iloc[:,2] == label].iloc[:,1].values[0])
#         print('---------------------------------------------------------------------')
#         print('rank: %s, label: %s, taxon: %s'%(str(0),label,taxon))
#         try:
#             img = np.moveaxis(image[0][0].asnumpy(), 0, 2)
#             pred_p = classifier_p(img[np.newaxis,:,:,:]).argmax()
#             if pred_p == label-label_offset[0]:
#                 prediction_label = global_mapping.loc[global_mapping['id'] == pred_p]['taxon'].values[0]
#
#                 explainer = lime_image.LimeImageExplainer(verbose=True)
#                 explanation = explainer.explain_instance(img, classifier_p, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
#                 plt.show()
#
#                 temp, mask = explanation.get_image_and_mask(label-label_offset[0], positive_only=True, num_features=superpixel_count, hide_rest=False)
#
#                 text_str = '\t' + 'p:  ' + prediction_label + ' - ' + str(pred_p)
#                 print(text_str)
#                 fn = str(i) + '__' + prediction_label
#
#                 outpath = os.path.join(path, fn)
#                 if not os.path.exists(outpath):
#                     os.mkdir(outpath)
#
#                 fn = str(i) + '__' + 'rank_' + str(0) + '__' + prediction_label + '.png'
#                 plt.imshow(skimage.segmentation.mark_boundaries(temp, mask, color=colors[1])/255)
#                 plt.text(2, 10, taxonomic_groups[0] + ' - explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
#                 plt.savefig(os.path.join(outpath,fn))
#                 plt.close()
#                 fn = str(i) + '__' + 'rank_' + str(5) + '__' + prediction_label + '_orig.png'
#                 plt.imshow(temp/255)
#                 plt.savefig(os.path.join(outpath,fn))
#                 plt.close()
#         except KeyError:
#             print('key erros: %s' % errors)
#             errors += 1
#
# ###################################################
# ####################  CLASS #######################
# ###################################################
# path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/c14'
# rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/c14/test'
# rec_prefix = 'class_test_oversampled_tt-split_SL'
# global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
# param_file = 'fun_per_lvl_tt-split_class_e14_f0.param'
#
# ############# load net ###############
# classes = 13
# gpus = mx.test_utils.list_gpus()
# ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
# pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
# c_finetune_net = get_model('densenet169', classes=classes)
# c_finetune_net.output.initialize(init.Xavier(), ctx=ctx)
# c_finetune_net.output.collect_params().setattr('lr_mult', 10)
# c_finetune_net.features = pretrained_net.features
# # finetune_net.collect_params().reset_ctx(ctx)
# c_finetune_net.hybridize()
# c_finetune_net.load_parameters(os.path.join(path,param_file))
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
# errors = 0
# for i,batch in enumerate(test_data):
#         label= gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
#         image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
#
#         taxon = str(global_mapping.loc[global_mapping.iloc[:,2] == label].iloc[:,1].values[0])
#         print('---------------------------------------------------------------------')
#         print('rank: %s, label: %s, taxon: %s'%(str(0),label,taxon))
#         try:
#             img = np.moveaxis(image[0][0].asnumpy(), 0, 2)
#             pred_c = classifier_c(img[np.newaxis,:,:,:]).argmax()
#             if pred_c == (label-label_offset[1]):
#                 prediction_label = global_mapping.loc[global_mapping['id'] == pred_c]['taxon'].values[0]
#
#                 explainer = lime_image.LimeImageExplainer(verbose=True)
#                 explanation = explainer.explain_instance(img, classifier_c, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
#                 plt.show()
#                 temp, mask = explanation.get_image_and_mask(label-label_offset[1], positive_only=True, num_features=superpixel_count, hide_rest=False)
#
#                 text_str = '\t' + 'p:  ' + prediction_label + ' - ' + str(pred_c)
#                 print(text_str)
#                 fn = str(i) + '__' + prediction_label
#
#                 outpath = os.path.join(path, fn)
#                 if not os.path.exists(outpath):
#                     os.mkdir(outpath)
#
#                 fn = str(i) + '__' + 'rank_' + str(2) + '__' + prediction_label + '.png'
#                 plt.imshow(skimage.segmentation.mark_boundaries(temp, mask, color=colors[1])/255)
#                 plt.text(2, 10, taxonomic_groups[2] + ' - explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
#                 plt.savefig(os.path.join(outpath,fn))
#                 plt.close()
#                 fn = str(i) + '__' + 'rank_' + str(5) + '__' + prediction_label + '_orig.png'
#                 plt.imshow(temp/255)
#                 plt.savefig(os.path.join(outpath,fn))
#                 plt.close()
#         except KeyError:
#             print('key erros: %s' % errors)
#             errors += 1

###################################################
####################  ORDER #######################
###################################################
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
o_finetune_net = get_model('densenet169', classes=classes)
o_finetune_net.output.initialize(init.Xavier(), ctx=ctx)
o_finetune_net.output.collect_params().setattr('lr_mult', 10)
o_finetune_net.features = pretrained_net.features
# finetune_net.collect_params().reset_ctx(ctx)
o_finetune_net.hybridize()
o_finetune_net.load_parameters(os.path.join(path,param_file))

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

errors = 0
for i,batch in enumerate(test_data):
    if i > 156:
        label= gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

        taxon = str(global_mapping.loc[global_mapping.iloc[:,2] == label].iloc[:,1].values[0])
        print('---------------------------------------------------------------------')
        print('rank: %s, label: %s, taxon: %s'%(str(0),label,taxon))
        try:
            img = np.moveaxis(image[0][0].asnumpy(), 0, 2)
            pred_o = classifier_o(img[np.newaxis,:,:,:]).argmax()
            if pred_o == (label-label_offset[2]):
                prediction_label = global_mapping.loc[global_mapping['id'] == pred_o]['taxon'].values[0]

                explainer = lime_image.LimeImageExplainer(verbose=True)
                explanation = explainer.explain_instance(img, classifier_o, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                plt.show()
                temp, mask = explanation.get_image_and_mask(label-label_offset[2], positive_only=True, num_features=superpixel_count, hide_rest=False)

                text_str = '\t' + 'p:  ' + prediction_label + ' - ' + str(pred_o)
                print(text_str)
                fn = str(i) + '__' + prediction_label

                outpath = os.path.join(path, fn)
                if not os.path.exists(outpath):
                    os.mkdir(outpath)

                fn = str(i) + '__' + 'rank_' + str(2) + '__' + prediction_label + '.png'
                plt.imshow(skimage.segmentation.mark_boundaries(temp, mask, color=colors[1])/255)
                plt.text(2, 10, taxonomic_groups[2] + ' - explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
                fn = str(i) + '__' + 'rank_' + str(5) + '__' + prediction_label + '_orig.png'
                plt.imshow(temp/255)
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
        except KeyError:
            print('key erros: %s' % errors)
            errors += 1

###################################################
#################### FAMILY #######################
###################################################
path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/f7'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/f7/test'
rec_prefix = 'family_test_oversampled_tt-split_SL'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_per_lvl_tt-split_family_e7_f0.param'

############# load net ###############
classes = 56
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
f_finetune_net = get_model('densenet169', classes=classes)
f_finetune_net.output.initialize(init.Xavier(), ctx=ctx)
f_finetune_net.output.collect_params().setattr('lr_mult', 10)
f_finetune_net.features = pretrained_net.features
# finetune_net.collect_params().reset_ctx(ctx)
f_finetune_net.hybridize()
f_finetune_net.load_parameters(os.path.join(path,param_file))

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

errors = 0
for i,batch in enumerate(test_data):
        label= gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

        taxon = str(global_mapping.loc[global_mapping.iloc[:,2] == label].iloc[:,1].values[0])
        print('---------------------------------------------------------------------')
        print('rank: %s, label: %s, taxon: %s'%(str(0),label,taxon))
        try:
            img = np.moveaxis(image[0][0].asnumpy(), 0, 2)
            pred_f = classifier_f(img[np.newaxis,:,:,:]).argmax()
            if pred_f == (label-label_offset[3]):
                prediction_label = global_mapping.loc[global_mapping['id'] == pred_f]['taxon'].values[0]

                explainer = lime_image.LimeImageExplainer(verbose=True)
                explanation = explainer.explain_instance(img, classifier_f, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                plt.show()
                temp, mask = explanation.get_image_and_mask(label-label_offset[3], positive_only=True, num_features=superpixel_count, hide_rest=False)

                text_str = '\t' + 'p:  ' + prediction_label + ' - ' + str(pred_f)
                print(text_str)
                fn = str(i) + '__' + prediction_label

                outpath = os.path.join(path, fn)
                if not os.path.exists(outpath):
                    os.mkdir(outpath)

                fn = str(i) + '__' + 'rank_' + str(3) + '__' + prediction_label + '.png'
                plt.imshow(skimage.segmentation.mark_boundaries(temp, mask, color=colors[1])/255)
                plt.text(2, 10, taxonomic_groups[3] + ' - explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
                fn = str(i) + '__' + 'rank_' + str(5) + '__' + prediction_label + '_orig.png'
                plt.imshow(temp/255)
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
        except KeyError:
            print('key erros: %s' % errors)
            errors += 1

###################################################
####################  GENUS #######################
###################################################
path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/g5'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/g5/test'
rec_prefix = 'genus_test_oversampled_tt-split_SL'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_per_lvl_tt-split_genus_e5_f0.param'

############# load net ###############
classes = 88
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
g_finetune_net = get_model('densenet169', classes=classes)
g_finetune_net.output.initialize(init.Xavier(), ctx=ctx)
g_finetune_net.output.collect_params().setattr('lr_mult', 10)
g_finetune_net.features = pretrained_net.features
# finetune_net.collect_params().reset_ctx(ctx)
g_finetune_net.hybridize()
g_finetune_net.load_parameters(os.path.join(path,param_file))

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

errors = 0
for i,batch in enumerate(test_data):
        label= gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

        taxon = str(global_mapping.loc[global_mapping.iloc[:,2] == label].iloc[:,1].values[0])
        print('---------------------------------------------------------------------')
        print('rank: %s, label: %s, taxon: %s'%(str(0),label,taxon))
        try:
            img = np.moveaxis(image[0][0].asnumpy(), 0, 2)
            pred_g = classifier_g(img[np.newaxis,:,:,:]).argmax()
            if pred_g == (label-label_offset[4]):
                prediction_label = global_mapping.loc[global_mapping['id'] == pred_g]['taxon'].values[0]

                explainer = lime_image.LimeImageExplainer(verbose=True)
                explanation = explainer.explain_instance(img, classifier_g, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                plt.show()
                temp, mask = explanation.get_image_and_mask(label-label_offset[4], positive_only=True, num_features=superpixel_count, hide_rest=False)

                text_str = '\t' + 'p:  ' + prediction_label + ' - ' + str(pred_g)
                print(text_str)
                fn = str(i) + '__' + prediction_label

                outpath = os.path.join(path, fn)
                if not os.path.exists(outpath):
                    os.mkdir(outpath)

                fn = str(i) + '__' + 'rank_' + str(4) + '__' + prediction_label + '.png'
                plt.imshow(skimage.segmentation.mark_boundaries(temp, mask, color=colors[1])/255)
                plt.text(2, 10, taxonomic_groups[4] + ' - explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
                fn = str(i) + '__' + 'rank_' + str(5) + '__' + prediction_label + '_orig.png'
                plt.imshow(temp/255)
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
        except KeyError:
            print('key erros: %s' % errors)
            errors += 1

###################################################
#################### SPECIES ######################
###################################################
path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/s14'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/SL_naiveOversampled_tt-split_SL_lr001_wd01/s14/test'
rec_prefix = 'species_test_oversampled_tt-split_SL'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_per_lvl_tt-split_species_e14_f0.param'

############# load net ###############
classes = 166
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
s_finetune_net = get_model('densenet169', classes=classes)
s_finetune_net.output.initialize(init.Xavier(), ctx=ctx)
s_finetune_net.output.collect_params().setattr('lr_mult', 10)
s_finetune_net.features = pretrained_net.features
# finetune_net.collect_params().reset_ctx(ctx)
s_finetune_net.hybridize()
s_finetune_net.load_parameters(os.path.join(path,param_file))

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

errors = 0
for i,batch in enumerate(test_data):
        label= gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

        taxon = str(global_mapping.loc[global_mapping.iloc[:,2] == label].iloc[:,1].values[0])
        print('---------------------------------------------------------------------')
        print('rank: %s, label: %s, taxon: %s'%(str(0),label,taxon))
        try:
            img = np.moveaxis(image[0][0].asnumpy(), 0, 2)
            pred_s = classifier_s(img[np.newaxis,:,:,:]).argmax()
            if pred_s == (label-label_offset[5]):
                prediction_label = global_mapping.loc[global_mapping['id'] == pred_s]['taxon'].values[0]

                explainer = lime_image.LimeImageExplainer(verbose=True)
                explanation = explainer.explain_instance(img, classifier_s, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                plt.show()
                temp, mask = explanation.get_image_and_mask(label-label_offset[5], positive_only=True, num_features=superpixel_count, hide_rest=False)

                text_str = '\t' + 'p:  ' + prediction_label + ' - ' + str(pred_s)
                print(text_str)
                fn = str(i) + '__' + prediction_label

                outpath = os.path.join(path, fn)
                if not os.path.exists(outpath):
                    os.mkdir(outpath)

                fn = str(i) + '__' + 'rank_' + str(5) + '__' + prediction_label + '.png'
                plt.imshow(skimage.segmentation.mark_boundaries(temp, mask, color=colors[1])/255)
                plt.text(2, 10, taxonomic_groups[5] + ' - explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
                fn = str(i) + '__' + 'rank_' + str(5) + '__' + prediction_label + '_orig.png'
                plt.imshow(temp/255)
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
        except KeyError:
            print('key erros: %s' % errors)
            errors += 1