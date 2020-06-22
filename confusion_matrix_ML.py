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

def plot_confusion_matrix(y_true,
                          y_pred,
                          target_names,
                          path,
                          title='Confusion matrix',
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

    cm = confusion_matrix(y_true, y_pred)

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
    plt.savefig(os.path.join(path, title+'.png'), bbox_inches='tight')
    plt.show()


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

phylum_labels = []
class_labels = []
order_labels = []
family_labels = []
genus_labels = []
species_labels = []

preds = dict()
phylum_taxon = set()
class_taxon  = set()
order_taxon  = set()
family_taxon  = set()
genus_taxon  = set()
species_taxon = set()
for i,batch in enumerate(test_data):
    phylum_label = gluon.utils.split_and_load(batch.label[0][:, 0], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    class_label = gluon.utils.split_and_load(batch.label[0][:, 1], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    order_label = gluon.utils.split_and_load(batch.label[0][:, 2], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    family_label = gluon.utils.split_and_load(batch.label[0][:, 3], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    genus_label = gluon.utils.split_and_load(batch.label[0][:, 4], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()
    species_label = gluon.utils.split_and_load(batch.label[0][:, 5], ctx_list=ctx, batch_axis=0, even_split=False)[0].asscalar()

    image = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)

    phylum_labels.append(phylum_label)
    class_labels.append(class_label)
    order_labels.append(order_label)
    family_labels.append(family_label)
    genus_labels.append(genus_label)
    species_labels.append(species_label)

    pred = finetune_net(image[0])

    for rank_idx in range(6):
        if rank_idx not in preds:
            preds[rank_idx] = []
        p = pred[rank_idx][0].asnumpy()
        preds[rank_idx].append(p.argmax()+label_offset[rank_idx])

cm_path = os.path.join(path,'cms')
if not os.path.exists(cm_path):
    os.mkdir(cm_path)

target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(phylum_labels)))]
plot_confusion_matrix(
    y_true=phylum_labels,
    y_pred=preds[0],
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Phylum',
    normalize=False)

# target_names = [mapping_df.loc[mapping_df.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(class_labels)))]
target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(set(class_labels))]
print(target_names)
plot_confusion_matrix(
    y_true=class_labels,
    y_pred=preds[1],
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Class',
    normalize=False)

target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(order_labels)))]
plot_confusion_matrix(
    y_true=order_labels,
    y_pred=preds[2],
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Order',
    normalize=False)

target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(family_labels)))]
plot_confusion_matrix(
    y_true=family_labels,
    y_pred=preds[3],
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Family',
    normalize=False)

target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(genus_labels)))]
plot_confusion_matrix(
    y_true=genus_labels,
    y_pred=preds[4],
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Genus',
    normalize=False)

target_names = [global_mapping.loc[global_mapping.iloc[:, 2] == l].iloc[:, 1].values[0] for l in list(sorted(set(species_labels)))]
plot_confusion_matrix(
    y_true=species_labels,
    y_pred=preds[5],
    # target_names=list(phylum_taxon),
    target_names=target_names,
    path=cm_path,
    title='Confusion_Matrix-Species',
    normalize=False)