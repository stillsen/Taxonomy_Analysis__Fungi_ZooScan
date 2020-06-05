import numpy as np
import mxnet as mx
from mxnet import init, nd, ndarray, gluon
from gluoncv.model_zoo import get_model
import skimage.io, skimage.segmentation
import matplotlib.pyplot as plt
from mxnet.gluon.data.vision import transforms
import os
from lime import lime_image
import time
from mxnet.io import ImageRecordIter
import pandas as pd
from ModelHandler import BigBangNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classifier_p(images):
    preds = finetune_net(transform(nd.array(images)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[0].asnumpy()
def classifier_c(images):
    preds = finetune_net(transform(nd.array(images)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[1].asnumpy()
def classifier_o(images):
    preds = finetune_net(transform(nd.array(images)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[2].asnumpy()
def classifier_f(images):
    preds = finetune_net(transform(nd.array(images)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[3].asnumpy()
def classifier_g(images):
    preds = finetune_net(transform(nd.array(images)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[4].asnumpy()
def classifier_s(images):
    preds = finetune_net(transform(nd.array(images)))
    # preds = finetune_net(transform(mx.nd.array(images[np.newaxis,:,:,:])))
    return preds[5].asnumpy()

path = '/home/stillsen/Documents/Data/Results/ExplainabilityPlot/ML-e11_orig_tt-split'
rec_path = '/home/stillsen/Documents/Data/Results/ExplainabilityPlot/ML-e11_orig_tt-split/all-in-one'
rec_prefix = 'all-in-one_orig_tt-split_ML_test'
param_file = 'fun_all-in-one_all-in-one_e11_f0.param'

mapping_df = pd.read_csv(os.path.join(rec_path, 'mapping.csv'))

############# load net ###############
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
finetune_net = BigBangNet(p=5,
                          c=11,
                          o=27,
                          f=46,
                          g=68,
                          s=166)
finetune_net.collect_params().initialize(init.Xavier(), ctx=ctx)
finetune_net.collect_params().setattr('lr_mult', 10)
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

c_permute = [[0,0,1],
             [0,1,0],
             [0,1,1],
             [1,0,0],
             [1,0,1],
             [1,1,0]]
avg_fit = 0
for i,batch in enumerate(test_data):
    label[0] = gluon.utils.split_and_load(batch.label[0][:, 0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    label[1] = gluon.utils.split_and_load(batch.label[0][:, 1], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    label[2] = gluon.utils.split_and_load(batch.label[0][:, 2], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    label[3] = gluon.utils.split_and_load(batch.label[0][:, 3], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    label[4] = gluon.utils.split_and_load(batch.label[0][:, 4], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    label[5] = gluon.utils.split_and_load(batch.label[0][:, 5], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()

    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

    mask = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None}
    for rank_idx in range(6):
        taxon = str(mapping_df.loc[mapping_df.iloc[:,2] == label[rank_idx]].iloc[:,1].values[0])
        print('label: %s, taxon: %s'%(label[rank_idx],taxon))
        outpath = os.path.join(path, 'l_' + str(label[rank_idx]) + '__' + taxon)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        prefix = os.path.join(outpath, 'l_' + str(label[rank_idx]) + '__' + taxon)
        try:
            img = np.moveaxis(image[0][0].asnumpy(),0,2).astype(float)/255
            explainer = lime_image.LimeImageExplainer(verbose=True)
            if rank_idx ==0:
                explanation = explainer.explain_instance(img, classifier_p, top_labels=5, hide_color=0, num_samples=1000)
            if rank_idx ==1:
                explanation = explainer.explain_instance(img, classifier_c, top_labels=5, hide_color=0, num_samples=1000)
            if rank_idx ==2:
                explanation = explainer.explain_instance(img, classifier_o, top_labels=5, hide_color=0, num_samples=1000)
            if rank_idx ==3:
                explanation = explainer.explain_instance(img, classifier_f, top_labels=5, hide_color=0, num_samples=1000)
            if rank_idx ==4:
                explanation = explainer.explain_instance(img, classifier_g, top_labels=5, hide_color=0, num_samples=1000)
            if rank_idx ==5:
                explanation = explainer.explain_instance(img, classifier_s, top_labels=5, hide_color=0, num_samples=1000)
            avg_fit += explanation.score

            temp, mask[rank_idx] = explanation.get_image_and_mask(label[rank_idx], positive_only=True, num_features=5, hide_rest=False)
            mask[rank_idx] *= rank_idx
        except KeyError:
            print('key erros: %s' % erros)
            erros += 1

    plt.imshow(temp)
    plt.savefig(prefix + '_' + str(i) + '.png')
    plt.show()
    plt.clf()

    counter = 1
    for rank_idx in range(6):
        if mask[rank_idx] is not None:
            plt.imshow(skimage.segmentation.mark_boundaries(temp, mask[rank_idx], color=c_permute[rank_idx]))
            # plt.show()
            # plt.clf()
            counter += 1
    plt.text(2, 10, 'explanation fit: ' + str(avg_fit/counter)[:4], color='white', fontsize=15, weight='bold')
    # plt.savefig(prefix + '_' + str(i) + '_explanation_cmb.png')
    plt.show()
    plt.clf()



print('key erros: %s'%erros)

