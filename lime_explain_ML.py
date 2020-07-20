import numpy as np
import mxnet as mx
from mxnet import init, nd, ndarray, gluon
from gluoncv.model_zoo import get_model
import skimage.io, skimage.segmentation
import matplotlib.pyplot as plt
from mxnet.gluon.data.vision import transforms
import os
import time
from mxnet.io import ImageRecordIter
import pandas as pd
from ModelHandler import BigBangNet
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# transform_tT = transforms.Compose([
#     transforms.ToTensor()
# ])

def classifier_p(images):
    # preds = finetune_net(transform_tT(nd.array(images)))
    preds = finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[0].asnumpy()
def classifier_c(images):
    # preds = finetune_net(transform_tT(nd.array(images)))
    preds = finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[1].asnumpy()
def classifier_o(images):
    # preds = finetune_net(transform_tT(nd.array(images)))
    preds = finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[2].asnumpy()
def classifier_f(images):
    # preds = finetune_net(transform_tT(nd.array(images)))
    preds = finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[3].asnumpy()
def classifier_g(images):
    # preds = finetune_net(transform_tT(nd.array(images)))
    preds = finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[4].asnumpy()
def classifier_s(images):
    # preds = finetune_net(transform_tT(nd.array(images)))
    preds = finetune_net(nd.array(np.moveaxis(images, 3, 1)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[5].asnumpy()

path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/ML_e4_naiveOversampled_tt-split_ML_lr001_wd01'
rec_path = '/home/stillsen/Documents/Data/Results_imv/ExplainabilityPlot/ML_e4_naiveOversampled_tt-split_ML_lr001_wd01/test'
rec_prefix = 'all-in-one_test_oversampled_tt-split_ML'
global_mapping = pd.read_csv(os.path.join('/home/stillsen/Documents/Data/Results_imv/global_mapping.csv'))
param_file = 'fun_all-in-one_all-in-one_e4_f0.param'

label_offset = [0, 6, 19, 50, 106, 194]
# best_scores_id = [8, 25, 30, 50, 62, 66, 83, 94, 95, 96, 101, 106, 114, 121, 136, 146, 152, 157, 171]
best_scores_id = [95, 96, 101, 106, 114, 121, 136, 146, 152, 157, 171]

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
    label_width=6,
    shuffle=False  # train true, test no
)

label = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None}
preds = dict()

erros = 0

# color permute
c_permute = [[0,0,1],
             [0,1,0],
             [0,1,1],
             [1,0,0],
             [1,0,1],
             [1,1,0]]
all_fits = []
# kernel_size=4
# max_dist=50
kernel_size=6
max_dist=50
ratio=.4
neighborhood_size = 1000
superpixel_count = 100


taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
taxonomic_groups_to_color = {'phylum': 0.857142857142857, 'class': 0.714285714285714, 'order': 0.571428571428571,
                             'family': 0.428571428571429, 'genus': 0.285714285714286, 'species': 0.142857142857143}

cc = [taxonomic_groups_to_color[x] for x in taxonomic_groups]
cmap = plt.cm.get_cmap(name='Dark2', lut=6)
colors = cmap(cc)
colors = colors[:,:3]

for i,batch in enumerate(test_data):
    if i in best_scores_id:
        label[0] = gluon.utils.split_and_load(batch.label[0][:, 0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        label[1] = gluon.utils.split_and_load(batch.label[0][:, 1], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        label[2] = gluon.utils.split_and_load(batch.label[0][:, 2], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        label[3] = gluon.utils.split_and_load(batch.label[0][:, 3], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        label[4] = gluon.utils.split_and_load(batch.label[0][:, 4], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
        label[5] = gluon.utils.split_and_load(batch.label[0][:, 5], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()

        image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

        mask = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None}
        print('#######################################################')
        for rank_idx in range(6):
            taxon = str(global_mapping.loc[global_mapping.iloc[:,2] == label[rank_idx]].iloc[:,1].values[0])
            print('---------------------------------------------------------------------')
            print('rank: %s, label: %s, taxon: %s'%(str(rank_idx),label[rank_idx],taxon))
            try:
                # img = np.moveaxis(image[0][0].asnumpy(),0,2).astype(float)/255
                img = np.moveaxis(image[0][0].asnumpy(), 0, 2)

                explainer = lime_image.LimeImageExplainer(verbose=True)
                if rank_idx ==0:
                    explanation = explainer.explain_instance(img, classifier_p, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                if rank_idx ==1:
                    explanation = explainer.explain_instance(img, classifier_c, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                if rank_idx ==2:
                    explanation = explainer.explain_instance(img, classifier_o, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                if rank_idx ==3:
                    explanation = explainer.explain_instance(img, classifier_f, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                if rank_idx ==4:
                    explanation = explainer.explain_instance(img, classifier_g, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                if rank_idx ==5:
                    explanation = explainer.explain_instance(img, classifier_s, top_labels=5, hide_color=0, num_samples=neighborhood_size,segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio))
                plt.show()
                all_fits.append(explanation.score)

                temp, mask[rank_idx] = explanation.get_image_and_mask(label[rank_idx]-label_offset[rank_idx], positive_only=True, num_features=superpixel_count, hide_rest=False)
                if rank_idx != 0:
                    mask[rank_idx] *= rank_idx

                pred_p = classifier_p(img[np.newaxis,:,:,:]).argmax()
                pred_c = classifier_c(img[np.newaxis,:,:,:]).argmax() + label_offset[1]
                pred_o = classifier_o(img[np.newaxis,:,:,:]).argmax() + label_offset[2]
                pred_f = classifier_f(img[np.newaxis,:,:,:]).argmax() + label_offset[3]
                pred_g = classifier_g(img[np.newaxis,:,:,:]).argmax() + label_offset[4]
                pred_s = classifier_s(img[np.newaxis,:,:,:]).argmax() + label_offset[5]

                phylum_prediction_label = global_mapping.loc[global_mapping['id'] == pred_p]['taxon'].values[0]
                class_prediction_label = global_mapping.loc[global_mapping['id'] == pred_c]['taxon'].values[0]
                order_prediction_label = global_mapping.loc[global_mapping['id'] == pred_o]['taxon'].values[0]
                family_prediction_label = global_mapping.loc[global_mapping['id'] == pred_f]['taxon'].values[0]
                genus_prediction_label = global_mapping.loc[global_mapping['id'] == pred_g]['taxon'].values[0]
                species_prediction_label = global_mapping.loc[global_mapping['id'] == pred_s]['taxon'].values[0]

                text_str = '\t' + 'p:  ' + phylum_prediction_label + ' - ' + str(pred_p) + '\n' + \
                           '\t' + 'c:  ' + class_prediction_label + ' - ' + str(pred_c) + '\n' + \
                           '\t' + 'o:  ' + order_prediction_label + ' - ' + str(pred_o) + '\n' + \
                           '\t' + 'f:  ' + family_prediction_label + ' - ' + str(pred_f) + '\n' + \
                           '\t' + 'g:  ' + genus_prediction_label + ' - ' + str(pred_g) + '\n' + \
                           '\t' + 's:  ' + species_prediction_label + ' - ' + str(pred_s)
                print(text_str)

                fn = str(i) + '__' + phylum_prediction_label + '_' + class_prediction_label + '_' + \
                     order_prediction_label + '_' + family_prediction_label + '_' + genus_prediction_label + \
                     '_' + species_prediction_label

                outpath = os.path.join(path, fn)
                if not os.path.exists(outpath):
                    os.mkdir(outpath)

                fn = str(i) + '__' + 'rank_' + str(rank_idx) + '__' + phylum_prediction_label + '_' + \
                     class_prediction_label + '_' + \
                     order_prediction_label + '_' + family_prediction_label + '_' + genus_prediction_label + \
                     '_' + species_prediction_label + '.png'
                print('\t color used'+str(c_permute[rank_idx]))
                plt.imshow(skimage.segmentation.mark_boundaries(temp, mask[rank_idx], color=colors[rank_idx])/255)
                plt.text(2, 10, taxonomic_groups[rank_idx] + ' - explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
                plt.savefig(os.path.join(outpath,fn))
                plt.close()
            except KeyError:
                print('key erros: %s' % erros)
                erros += 1

        marked_img = temp
        for rank_idx in range(6):
            if mask[rank_idx] is not None:
                marked_img = skimage.segmentation.mark_boundaries(marked_img, mask[rank_idx], color=colors[rank_idx])
        plt.imshow(marked_img/255)
        plt.text(2, 10, 'explanation fit: ' + str(np.mean(all_fits))[:4], color='white', fontsize=15, weight='bold')
        fn = str(i) + '__' + phylum_prediction_label + '_' + \
             class_prediction_label + '_' + \
             order_prediction_label + '_' + family_prediction_label + '_' + genus_prediction_label + \
             '_' + species_prediction_label + '.png'
        plt.savefig(os.path.join(outpath,fn))
        plt.close()

print('key erros: %s'%erros)

