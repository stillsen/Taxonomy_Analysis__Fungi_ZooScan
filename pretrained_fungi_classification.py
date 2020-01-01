import mxnet as mx
import numpy as np
import os, time, shutil, re, argparse
import pandas as pd
import seaborn as sns

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model
from multiprocessing import cpu_count
from matplotlib import pyplot as plt

# load custom modules
from DataHandler import DataHandler
from DataPrep import DataPrep


# return metrics string representation
def metric_str(names, accs):
    return ', '.join(['%s=%f'%(names, accs) ])
    # return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])

def evaluate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()


def train(net, train_iter, val_iter, epochs, ctx, metric, lr, momentum, wd, batch_size, path, taxa = '', log_interval = 10):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    num_batch = len(train_iter)
    # num_batch = 1
    print('batch num: %d'%num_batch)
    best_acc = 0

    val_names, val_accs = evaluate(net, val_iter, ctx)
    print('[Initial] validation: %s'%(metric_str(val_names, val_accs)))
    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_loss = 0
        metric.reset()

        for i, batch in enumerate(train_iter):
            # the model zoo models expect normalized images
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]
                for l in loss:
                    l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)

            if log_interval and not (i+1)%log_interval:
                names, accs = metric.get()
                # print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                #                epoch, i, batch_size/(time.time()-btic), metric_str(names, accs)))
            btic = time.time()

        names, accs = metric.get()
        metric.reset()
        print('[Epoch %d] training: %s'%(epoch, metric_str(names, accs)))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        val_names, val_accs = evaluate(net, val_iter, ctx)
        print('[Epoch %d] validation: %s'%(epoch, metric_str(val_names, val_accs)))
        train_loss /= num_batch


        if val_accs > best_acc:
            best_acc = val_accs
            print('Best validation acc found. Checkpointing...')
            net.save_parameters(os.path.join(path, 'deep-fungi-%s-%d.params'%(taxa,epoch)))
    return net

# def parse_opts():
#     parser = argparse.ArgumentParser(description='Preparing MINC 2500 Dataset',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--data', type=str, required=True,
#                         help='directory for the original data folder')
#     opts = parser.parse_args()
#     return opts


# Preparation
# opts = parse_opts()

# path = opts.data
path = "/home/stillsen/Documents/Data/Image_classification_soil_fungi__working_copy"

## PARAMETERS


epochs = 10
lr = 0.001
per_device_batch_size = 1
momentum = 0.9
wd = 0.0001

# lr_factor = 0.75
# lr_steps = [10, 20, 30, np.inf]

num_gpus = 1
num_workers = cpu_count()

gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]

batch_size = per_device_batch_size * max(num_gpus, 1)

# AUGMENTATION
jitter_param = 0.4
lighting_param = 0.1
# Metric
metric = mx.metric.Accuracy()


missing_values = ['', 'unknown', 'unclassified']
csv_path = os.path.join(path, 'im_merged.csv')
df = pd.read_csv(csv_path, na_values=missing_values)

print('NaNs in the label data set')
print(df.isnull().sum())

taxonomic_groups = ['phylum', 'class', 'order', 'family', 'genus', 'species']
# taxonomic_groups = ['phylum']
fig, ax = plt.subplots(2, 3, sharey='row')
sns.set_context("paper")
for (i, taxa) in enumerate(taxonomic_groups):
    print('working in taxonomic group: %s' %taxa)
    DataPrep(taxa=taxa, path=path, dataset='fungi', df=df)
    data_handler = DataHandler(path=path,
                               batch_size = batch_size,
                               num_workers=num_workers,
                               transform = True)
    classes = data_handler.classes

    # plot class distribution
    # x = []
    # y = []
    # for item in enumerate(data_handler.samples_per_class):
    #     x = item[0]
    #     y = item[1]
    x = list(data_handler.samples_per_class.keys())
    y = list(data_handler.samples_per_class.values())
    # x, y = zip(data_handler.samples_per_class.items())
    if i < 3:
        # ax[0][i].plot(x,y)
        ax[0][i].xaxis.set_visible(False)
        sns.barplot(x=x, y=y, color="b", ax=ax[0][i])
        # ax[0][i].hist(y)
        ax[0][i].set_title(taxa)
    else:
        ax[1][i - 3].xaxis.set_visible(False)
        sns.barplot(x=x,y=y, color="b", ax=ax[1][i - 3])
        # ax[1][i - 3].hist(y)
        ax[1][i - 3].set_title(taxa)
        # ax[1][i-3].plot(x, y)
    for cl in data_handler.samples_per_class:
        print("not resampled %s --- %s: %d"%(taxa,cl,data_handler.samples_per_class[cl]))
        print("    resampled %s --- %s: %d" % (taxa, cl, data_handler.samples_per_class_normalized[cl]))
    ## Model Densenet 169
    model_name = 'densenet169'
    finetune_net = get_model(model_name, pretrained=True)
    with finetune_net.name_scope():
        finetune_net.output = nn.Dense(classes)
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    ### load parameters if already trained, otherwise train
    model_loaded = False
    param_file = ''
    e = -1
    for file_name in os.listdir(path):
        if re.match('deep-fungi-%s-'% taxa, file_name):
            if int(file_name.split('-')[-1][0]) > e:
                e = int(file_name.split('-')[-1][0])
                param_file = os.path.join(path, file_name)
            model_loaded = True
    if not model_loaded: # train
        print('training model for %s' % taxa)
        finetune_net= train(finetune_net,
                            data_handler.train_data,
                            data_handler.test_data,
                            epochs=epochs,
                            ctx=ctx,
                            metric=metric,
                            lr=lr,
                            momentum=momentum,
                            wd=wd,
                            batch_size=batch_size,
                            taxa=taxa,
                            path=path)
    else:
        finetune_net.load_parameters(param_file)
        print('loading %s' %param_file)
    val_names, val_accs = evaluate(finetune_net, data_handler.test_data, ctx)
    print('%s: %s' % (taxa, metric_str(val_names, val_accs)))



plt.show()
