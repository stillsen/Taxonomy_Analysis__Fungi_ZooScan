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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classifier_fn(images):
    preds = finetune_net(transform(nd.array(images)))
    return preds.asnumpy()

path = '/home/stillsen/Documents/Data/Results/ExplainabilityPlot/SL-class-e11_naiveOversampled_tt-split/'
rec_path = '/home/stillsen/Documents/Data/Results/ConfusionMatrix/SL-class-e11_naiveOversampled_tt-split/class/test'
rec_prefix = 'class_test_oversampled_tt-split_SL'
param_file = 'fun_per_lvl_tt-split_class_e11_f0.param'

classes = 11
mapping_df = pd.read_csv(os.path.join(rec_path, 'mapping.csv'))

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
    shuffle=False
)

gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
finetune_net = get_model('densenet169', classes=classes)
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
finetune_net.output.collect_params().setattr('lr_mult', 10)
finetune_net.features = pretrained_net.features
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()
finetune_net.load_parameters(os.path.join(path, param_file))

erros = 0
for i,batch in enumerate(test_data):
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = label[0].asscalar()
    taxon = mapping_df.loc[mapping_df['id'] == label]['taxon'].values[0]
    print('label: %s, taxon: %s'%(label,taxon))
    # skimage.io.imshow(image[0][0].asnumpy().transpose().astype(int))
    # plt.show()
    try:
        img = np.moveaxis(image[0][0].asnumpy(),0,2).astype(float)/255
        explainer = lime_image.LimeImageExplainer(verbose=True)
        explanation = explainer.explain_instance(img, classifier_fn,top_labels=5, hide_color=0, num_samples=1000)

        outpath = os.path.join(path, 'l_' + str(label) + '__' + taxon)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        prefix = os.path.join(outpath, 'l_' + str(label) + '__' + taxon)

        temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)
        plt.imshow(temp)
        plt.savefig(prefix + '_' + str(i) + '.png')
        plt.clf()
        plt.imshow(skimage.segmentation.mark_boundaries(temp, mask))
        plt.text(2, 10, 'explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
        plt.savefig(prefix + '_' + str(i) + '_explanation_pos.png')
        plt.clf()
        # plt.show()
        temp, mask = explanation.get_image_and_mask(label, positive_only=False, num_features=10, hide_rest=False)
        plt.imshow(skimage.segmentation.mark_boundaries(temp, mask))
        plt.text(2, 10, 'explanation fit: ' + str(explanation.score)[:4], color='white', fontsize=15, weight='bold')
        plt.savefig(prefix + '_' + str(i) + '_explanation_all.png')
        plt.clf()
    except KeyError:
        print('key erros: %s' % erros)
        erros += 1
print('key erros: %s'%erros)