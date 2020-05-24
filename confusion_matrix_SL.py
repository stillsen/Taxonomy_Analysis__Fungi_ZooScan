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

def plot_confusion_matrix(y_true,
                          y_pred,
                          target_names,
                          path,
                          title='Confusion matrix - SL Class',
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

    cm = confusion_matrix(labels, preds)

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
        plt.xticks(tick_marks, target_names, rotation=45)
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
    plt.savefig(os.path.join(path,'confusionmatrix.png'), bbox_inches='tight')
    plt.show()



transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# path = '/home/stillsen/Documents/Data/Results/ConfusionMatrix/SL-class-e11_naiveOversampled_tt-split/'
# param_file = os.path.join(path,'fun_per_lvl_tt-split_class_e11_f0.param')
# path = '/home/stillsen/Documents/Data/Results/ConfusionMatrix/SL-class-e11_naiveOversampled_tt-split/class/test'
root_path = '/home/stillsen/Documents/Data/Results/ExplainabilityPlot/SL-class-e11_naiveOversampled_tt-split/'
param_file = os.path.join(root_path,'fun_per_lvl_tt-split_class_e11_f0.param')
path = '/home/stillsen/Documents/Data/Results/ExplainabilityPlot/SL-class-e11_naiveOversampled_tt-split/class/test'
############# load net ###############
classes = 11
gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]
pretrained_net = get_model('densenet169', pretrained=True, ctx=ctx)
finetune_net = get_model('densenet169', classes=classes)
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
finetune_net.output.collect_params().setattr('lr_mult', 10)
finetune_net.features = pretrained_net.features
# finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()
finetune_net.load_parameters(param_file)

############# load data ##############
test_data = ImageRecordIter(
    path_imgrec=os.path.join(path,'class_test_oversampled_tt-split_SL.rec'),
    path_imgidx=os.path.join(path,'class_test_oversampled_tt-split_SL.idx'),
    path_imglist=os.path.join(path,'class_test_oversampled_tt-split_SL.lst'),
    aug_list=mx.image.CreateAugmenter((3, 224, 224), inter_method=1),
    data_shape=(3, 224, 224),
    batch_size=1,
    resize=224,
    label_width=1,
    shuffle=False  # train true, test no
)

labels = []
preds = []
for batch in test_data:

    # skimage.io.imshow(image.data[0][0].asnumpy().transpose().astype(int))
    # plt.show()
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
    labels.append(label[0].asscalar())
    # loop over folder with test data, for every image collect label and prediction
    #
    # np.random.seed(222)
    # # print(Xi[np.newaxis,:,:,:])
    # preds = finetune_net(transform(mx.nd.array(Xi[np.newaxis,:,:,:])))
    pred = finetune_net(image[0])
    # # top_pred_classes = preds[0].argsort()[-5:][::-1]
    pred = pred.argmax(axis=1).asscalar()
    preds.append(pred)
    # print('observation prediction: %s' %top_pred_classes)
    #
df = pd.read_csv(os.path.join(path,'mapping.csv'), header=0,sep=',')
target_names = df['taxon']

plot_confusion_matrix(y_true=labels, y_pred=preds, target_names=target_names, path=root_path, normalize=False)
