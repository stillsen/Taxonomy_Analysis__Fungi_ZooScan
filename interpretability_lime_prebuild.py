import numpy as np
import mxnet as mx
from mxnet import init, nd, ndarray
from gluoncv.model_zoo import get_model
import skimage.io, skimage.segmentation, copy, sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mxnet.gluon.data.vision import transforms
import os
from lime import lime_image
import time

transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classifier_fn(images):
    preds = finetune_net(transform(nd.array(images)))
    return preds.asnumpy()

path = '/home/stillsen/Documents/Data/Results/ExplainabilityPlot/SL-class-e11_naiveOversampled_tt-split/'
image_folder = os.path.join(path,'Sordariomycetes_test')
file_prefix = '21_10_19_C__21.10.19_C__cut__9'
# file_prefix = 'MOM_EX_010_D__27.02.18_D__cut__6'
# file_prefix = 'MOM_EX_014_B__14.03.18_B__cut__5'
# file_prefix = 'EKA_4.12.17_H__4.12.17_H__cut__1'
classes = 11

gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
finetune_net = get_model('densenet169', classes=classes)
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
finetune_net.output.collect_params().setattr('lr_mult', 10)
finetune_net.features = pretrained_net.features
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()
finetune_net.load_parameters(os.path.join(path,'fun_per_lvl_tt-split_class_e11_f0.param'))

Xi = skimage.io.imread(os.path.join(image_folder, file_prefix+'.png'))
Xi = skimage.transform.resize(Xi, (224,224))

explainer = lime_image.LimeImageExplainer(verbose=True)

tmp = time.time()
# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
# explanation = explainer.explain_instance(Xi, classifier_fn, top_labels=5, hide_color=0, num_samples=1000)
explanation = explainer.explain_instance(Xi, classifier_fn, top_labels=5, hide_color=0, num_samples=1000)
print(time.time() - tmp)

# temp, mask = explanation.get_image_and_mask(7, positive_only=True, num_features=5, hide_rest=True)
# plt.imshow(skimage.segmentation.mark_boundaries(temp, mask))
# plt.show()
skimage.io.imsave(os.path.join(path, file_prefix+'.png'), Xi)

temp, mask = explanation.get_image_and_mask(7, positive_only=True, num_features=5, hide_rest=False)
plt.imshow(skimage.segmentation.mark_boundaries(temp, mask))
plt.savefig(os.path.join(path,file_prefix+'_explanation_pos.png'))
plt.show()

temp, mask = explanation.get_image_and_mask(7, positive_only=False, num_features=20, hide_rest=False)
plt.imshow(skimage.segmentation.mark_boundaries(temp, mask))
plt.savefig(os.path.join(path,file_prefix+'_explanation_all.png'))
plt.show()
