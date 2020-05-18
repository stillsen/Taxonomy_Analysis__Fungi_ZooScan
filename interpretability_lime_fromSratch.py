import numpy as np
import mxnet as mx
from mxnet import init
from gluoncv.model_zoo import get_model
import skimage.io, skimage.segmentation, copy, sklearn, mxnet.ndarray
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mxnet.gluon.data.vision import transforms
import os


transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

path = '/home/stillsen/Documents/Data/Results/ExplainabilityPlot/SL-class-e11_naiveOversampled_tt-split/'
image_folder = os.path.join(path,'Sordariomycetes_test')
# file_prefix = 'MOM_EX_010_D__27.02.18_D__cut__6'
# file_prefix = 'MOM_EX_014_B__14.03.18_B__cut__5'
file_prefix = 'EKA_4.12.17_H__4.12.17_H__cut__1'
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
# Xi = skimage.transform.resize(Xi, (299,299))
Xi = skimage.transform.resize(Xi, (224,224))
# Xi = (Xi - 0.5)*2 #Inception pre-processing
# skimage.io.imshow(Xi/2+0.5) # Show image before inception preprocessing
# skimage.io.imshow(Xi)
# plt.show()

np.random.seed(222)
# print(Xi[np.newaxis,:,:,:])
preds = finetune_net(transform(mx.nd.array(Xi[np.newaxis,:,:,:])))
# preds = finetune_net(mx.nd.array(Xi))
# top_pred_classes = preds[0].argsort()[-5:][::-1]
top_pred_classes = preds.asnumpy().argmax(axis=1)
print('observation prediction: %s' %top_pred_classes)


# superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
superpixels = skimage.segmentation.quickshift(Xi, kernel_size=2,max_dist=5, ratio=0.2)
num_superpixels = np.unique(superpixels).shape[0]

skimage.io.imsave(os.path.join(path, file_prefix+'.png'), Xi)
skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi, superpixels))
skimage.io.imsave(os.path.join(path, file_prefix+'_segmentation.png'), skimage.segmentation.mark_boundaries(Xi, superpixels))
plt.show()
num_perturb = 150
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

def perturb_image(img,perturbation,segments):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image

skimage.io.imshow(perturb_image(Xi,perturbations[0],superpixels))
# skimage.io.imshow(perturbations[0])
# plt.show()

predictions = []
for pert in perturbations:
  perturbed_img = perturb_image(Xi,pert,superpixels)
  # pred = finetune_net(transform(mx.nd.array(perturbed_img[np.newaxis,:,:,:])))
  pred = finetune_net(transform(mx.nd.array(perturbed_img[np.newaxis, :, :, :]))).asnumpy()
  # pred = pred.asnumpy().argmax(axis=1)
  predictions.append(pred)

predictions = np.array(predictions)
# print('pertubation predictions: %s' %predictions)
print('pertubation prediction shape: %s' %(predictions.shape[0]))

original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
print(distances.shape)

kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
print(weights.shape)

class_to_explain = int(top_pred_classes[0])
print(class_to_explain)
# class_to_explain = top_pred_classes[0]
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]

num_top_features = 4
top_features = np.argsort(coeff)[-num_top_features:]

mask = np.zeros(num_superpixels)
mask[top_features]= True #Activate top superpixels
skimage.io.imshow(perturb_image(Xi,mask,superpixels) )
skimage.io.imsave(os.path.join(path,file_prefix+'_explanation.png'), perturb_image(Xi,mask,superpixels))
plt.show()