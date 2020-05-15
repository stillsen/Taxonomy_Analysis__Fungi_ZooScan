import os, time
import mxnet as mx
from mxnet import gluon, init, gpu, nd
from mxnet import autograd as ag
from mxnet.ndarray import concat
from gluoncv.model_zoo import get_model
import pandas as pd

from mxnet.ndarray.contrib import isnan

class BigBangNet(gluon.HybridBlock):
    def __init__(self, p, c, o , f, g, s):
        super(BigBangNet, self).__init__()
        # self.params.get()
        with self.name_scope():

            self.feature = None

            self.phylum_out = gluon.nn.Dense(p)
            self.class_out   = gluon.nn.Dense(c)
            self.order_out = gluon.nn.Dense(o)
            self.family_out = gluon.nn.Dense(f)
            self.genus_space = gluon.nn.Dense(g)
            self.species_space = gluon.nn.Dense(s)

    def hybrid_forward(self, F, x):

        featureX = self.feature(x)

        out1 = self.phylum_out(featureX)
        out2 = self.class_out(featureX)
        out3 = self.order_out(featureX)
        out4 = self.family_out(featureX)
        out5 = self.genus_space(featureX)
        out6 = self.species_space(featureX)

        return (out1, out2, out3, out4, out5, out6)


class ModelHandler:
    def __init__(self,
                 classes,
                 # epochs = 10,
                 metrics,
                 rank_idx=None,
                 learning_rate = 0.001,
                 batch_size = 1,
                 momentum = 0.9,
                 wd=0.0001,
                 num_gpus=1,
                 num_workers=1,
                 model_name = 'densenet169',
                 prior_param = None,
                 multi_label_lvl = 1,
                 ):
        ##
        # multilabel_lvl = 1 --> separate local level classifiers
        # multilabel_lvl = 2 --> big bang classifiers
        # multilabel_lvl = 3 --> chained local level classifiers
        ##
        # self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.wd = wd
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.metrics = metrics
        self.best_model = None

        self.gpus = mx.test_utils.list_gpus()
        self.ctx = [mx.gpu(i) for i in self.gpus] if len(self.gpus) > 0 else [mx.cpu()]
        # self.ctx = gpu()

        self.multi_label_lvl = multi_label_lvl
        self.rank = rank_idx

        self.net = self.setup_net(multi_label_lvl=multi_label_lvl,
                                  model_name=model_name,
                                  classes=classes,
                                  ctx=self.ctx)


    def setup_net(self, multi_label_lvl, model_name, classes, ctx):
        finetune_net = None
        if multi_label_lvl == 1: #separate local level classifiers
            pretrained_net = get_model(model_name, pretrained=True, ctx=ctx)
            finetune_net = get_model(model_name, classes=classes)
            finetune_net.output.initialize(init.Xavier(), ctx=ctx)
            finetune_net.output.collect_params().setattr('lr_mult', 10)
            finetune_net.features = pretrained_net.features
            finetune_net.collect_params().reset_ctx(ctx)
            finetune_net.hybridize()
            self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        if multi_label_lvl == 2: # all-in-one/big bang classifiers
            pretrained_net = get_model(model_name, pretrained=True, ctx=ctx)
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
            self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


        if multi_label_lvl == 3: # hierarchical/chained local level classifiers
            pretrained_net = get_model(model_name, pretrained=True, ctx=ctx)
            finetune_net = get_model(model_name, classes=classes)
            finetune_net.output.initialize(init.Xavier(), ctx=ctx)
            finetune_net.output.collect_params().setattr('lr_mult', 10)
            finetune_net.features = pretrained_net.features
            finetune_net.collect_params().reset_ctx(ctx)
            finetune_net.hybridize()
            self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        return finetune_net

    def add_layer(self, prior_param, rank_idx, classes):
        # updates rank_idx and loads the parameters for the best previous net and adds a new layer
        ################
        self.rank = rank_idx
        self.net.load_parameters(prior_param)
        net = gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(self.net)
            net.add(gluon.nn.Dense(classes))
        # initialize the parameters
        net.collect_params().initialize()
        # finetune_net.output.initialize(init.Xavier(), ctx=ctx)
        net.collect_params().setattr('lr_mult', 10)
        net.collect_params().reset_ctx(self.ctx)
        net.hybridize()
        self.net = net

    def metric_str(self, names, accs):
        # return metrics string representation
        return ', '.join(['%s=%f'%(names, accs) ])
        # return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])

    def evaluate(self, net, val_data, ctx, metric):
        # metric = mx.metric.Accuracy()
        for batch in val_data:
            # data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            data =  gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
            # label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            # if self.multi_label_lvl == 2 or self.multi_label_lvl == 3:
            if self.multi_label_lvl == 3:
                label = gluon.utils.split_and_load(batch.label[0][:,self.rank], ctx_list=ctx, batch_axis=0, even_split=False)
            else:
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
            outputs = [net(X) for X in data]
            if self.multi_label_lvl == 2 :
                for i, o in enumerate(outputs[0]):
                    l = nd.slice_axis(label[0], axis=1, begin=i, end=i + 1)
                    metric[i].update(preds=o, labels=l)
                # print(metric.get()[1])
            else:
                metric.update(label, outputs)
        val_data.reset()
        return metric

    def _save_all(self, ext_storage_path, param_file_name, net, epoch, fold):
        print('\t\t\tsaving parameters')
        e_param_file_name = param_file_name.split('.')[0]+'_e'+str(epoch)+'_f'+str(fold)+'.param'
        abs_path_param_file_name = os.path.join(ext_storage_path, e_param_file_name)
        net.save_parameters(abs_path_param_file_name)



    def train(self, train_iter, val_iter, epochs, param_file_name, fold, start_epoch=0, load_param_file = None, ext_storage_path=''):
        net = self.net
        ctx = self.ctx
        metric = self.metrics
        lr = self.learning_rate
        momentum = self.momentum
        wd = self.wd
        batch_size = self.batch_size
        loss_fn = self.loss_fn

        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})

        if self.multi_label_lvl == 2:
            metric = [mx.metric.PCC(), mx.metric.PCC(), mx.metric.PCC(), mx.metric.PCC(), mx.metric.PCC(), mx.metric.PCC()]
            acc_metric = [mx.metric.Accuracy(), mx.metric.Accuracy(), mx.metric.Accuracy(), mx.metric.Accuracy(), mx.metric.Accuracy(), mx.metric.Accuracy()]
            list_train_score = { 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus':  [], 'species': []}
            list_val_score = { 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus':  [], 'species': []}
            list_train_acc = { 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus':  [], 'species': []}
            list_val_acc = { 'phylum': [], 'class': [], 'order': [], 'family': [], 'genus':  [], 'species': []}
            list_epochs = []
            metric = self.evaluate(net, val_iter, ctx, metric)
            acc_metric = self.evaluate(net, val_iter, ctx, acc_metric)
            score_val = []
            acc_val = []
            print('\tComputing initial validation score...')
            for i in range(6):
                val_names, sv = metric[i].get()
                score_val.append(sv)
                acc_names, av = acc_metric[i].get()
                acc_val.append(av)
                print('\t[Initial] validation: [Rank %i] validation: %s %s' % ( i, self.metric_str(val_names, score_val[i]), self.metric_str(acc_names, acc_val[i])))
        else:
            print('\tComputing initial validation score...')
            val_names, score_val = self.evaluate(net, val_iter, ctx, metric=self.metrics).get()
            print('\t[Initial] validation: %s' % (self.metric_str(val_names, score_val)))
            acc_metric = mx.metric.Accuracy()
            list_train_score = []
            list_val_score = []
            list_train_acc = []
            list_val_acc = []
            list_epochs = []
        if load_param_file is not None:
            net.load_parameters(load_param_file)
        for epoch in range(start_epoch, epochs):
            prev_score_val = 0
            tic = time.time()
            btic = time.time()
            if self.multi_label_lvl == 2:
                for i in range(6):
                    metric[i].reset()
                    acc_metric[i].reset()
            else:
                metric.reset()
                acc_metric.reset()

            for batch in train_iter:
                # print('training... epoch:  %d  batch_no:  %s' %(epoch,i))
                # the model zoo models expect normalized images
                # data = gluon.utils.split_and_load(batch.data, ctx_list=ctx, batch_axis=0, even_split=False)
                # label = gluon.utils.split_and_load(batch.label, ctx_list=ctx, batch_axis=0, even_split=False)
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
                if self.multi_label_lvl == 3:
                    label = gluon.utils.split_and_load(batch.label[0][:, self.rank], ctx_list=ctx, batch_axis=0, even_split=False)
                else:
                    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
                if self.multi_label_lvl == 2:
                    l0 = list()
                    l1 = list()
                    l2 = list()
                    l3 = list()
                    l4 = list()
                    l5 = list()
                    o0 = list()
                    o1 = list()
                    o2 = list()
                    o3 = list()
                    o4 = list()
                    o5 = list()
                    l = nd.slice_axis(label[0], axis=1, begin=0, end=1)
                    l0.append(l.reshape(len(l)))
                    l = nd.slice_axis(label[0], axis=1, begin=1, end=2)
                    l1.append(l.reshape(len(l)))
                    l = nd.slice_axis(label[0], axis=1, begin=2, end=3)
                    l2.append(l.reshape(len(l)))
                    l = nd.slice_axis(label[0], axis=1, begin=3, end=4)
                    l3.append(l.reshape(len(l)))
                    l = nd.slice_axis(label[0], axis=1, begin=4, end=5)
                    l4.append(l.reshape(len(l)))
                    l = nd.slice_axis(label[0], axis=1, begin=5, end=6)
                    l5.append(l.reshape(len(l)))
                    with ag.record():
                        outputs = [net(X) for X in data]
                        o0.append(outputs[0][0])
                        o1.append(outputs[0][1])
                        o2.append(outputs[0][2])
                        o3.append(outputs[0][3])
                        o4.append(outputs[0][4])
                        o5.append(outputs[0][5])
                        loss0=[]
                        loss1=[]
                        loss2=[]
                        loss3=[]
                        loss4=[]
                        loss5=[]
                        loss=[]
                        for yhat, y in zip(o0, l0):
                            loss0 = [*loss0, loss_fn(yhat, y)]
                        for yhat, y in zip(o1, l1):
                            loss1 = [*loss1, loss_fn(yhat, y)]
                        for yhat, y in zip(o2, l2):
                            loss2 = [*loss2, loss_fn(yhat, y)]
                        for yhat, y in zip(o3, l3):
                            loss3 = [*loss3, loss_fn(yhat, y)]
                        for yhat, y in zip(o4, l4):
                            loss4 = [*loss4, loss_fn(yhat, y)]
                        for yhat, y in zip(o5, l5):
                            loss5 = [*loss5, loss_fn(yhat, y)]
                        loss.append(concat(loss0[0],loss1[0],loss2[0],loss3[0],loss4[0],loss5[0],dim=0))
                        for l in loss:
                            l.backward()
                    trainer.step(batch_size)
                    #####################
                    for i, o in enumerate(outputs[0]):
                        l = nd.slice_axis(label[0], axis=1, begin=i, end=i + 1)
                        metric[i].update(preds=o, labels=l)
                        acc_metric[i].update(preds=o, labels=l)
                    #####################
                    # metric.update(label, outputs)
                    # acc_metric.update(label, outputs)
                else:
                    with ag.record():
                        outputs = [net(X) for X in data]
                        loss=[]
                        for yhat, y in zip(outputs, label):
                            loss = [*loss, loss_fn(yhat, y)]
                        for l in loss:
                            l.backward()
                    trainer.step(batch_size)
                    metric.update(label, outputs)
                    acc_metric.update(label, outputs)
            if self.multi_label_lvl == 2:
                score_train = []
                acc_train = []
                score_val = []
                acc_val = []
                for i in range(6):
                    name, st = metric[i].get()
                    score_train.append(st)
                    acc_name, at = acc_metric[i].get()
                    acc_train.append(at)
                    metric[i].reset()
                    acc_metric[i].reset()
                    print('\t[Fold %d Epoch %d] [Rank %i] training: %s %s' % (fold, epoch, i, self.metric_str(name, score_train[i]), self.metric_str(acc_name, acc_train[i])))
                    # print('\t[Fold %d Epoch %d] time cost: %f' % (fold, epoch, time.time() - tic))
                metric = self.evaluate(net, val_iter, ctx, metric)
                acc_metric = self.evaluate(net, val_iter, ctx, acc_metric)
                for i in range(6):
                    val_names, sv = metric[i].get()
                    score_val.append(sv)
                    acc_names, av = acc_metric[i].get()
                    acc_val.append(av)
                    print('\t[Fold %d Epoch %d] [Rank %i] validation: %s %s' % (fold, epoch, i, self.metric_str(val_names, score_val[i]), self.metric_str(acc_names, acc_val[i])))
                    # print('\t[Fold %d Epoch %d] validation: %s' % (fold, epoch, self.metric_str(acc_name, acc_val)))
                train_iter.reset()
            else:
                name, score_train = metric.get()
                acc_name, acc_train = acc_metric.get()
                metric.reset()
                acc_metric.reset()
                train_iter.reset()
                print('\t[Fold %d Epoch %d] training: %s'%(fold, epoch, self.metric_str(name, score_train)))
                print('\t[Fold %d Epoch %d] training: %s' % (fold, epoch, self.metric_str(acc_name, acc_train)))
                print('\t[Fold %d Epoch %d] time cost: %f'%(fold, epoch, time.time()-tic))
                val_names, score_val = self.evaluate(net, val_iter, ctx, self.metrics).get()
                acc_names, acc_val = self.evaluate(net, val_iter, ctx, acc_metric).get()
                print('\t[Fold %d Epoch %d] validation: %s'%(fold, epoch, self.metric_str(val_names, score_val)))
                print('\t[Fold %d Epoch %d] validation: %s' % (fold, epoch, self.metric_str(acc_name, acc_val)))
                if score_val > prev_score_val:
                    prev_score_val = score_val
                    self.best_model = param_file_name.split('.')[0]+'_e'+str(epoch)+'_f'+str(fold)+'.param'


            # ext_storage_path, param_file_name, app_file_name, net_list, score_list, app
            self._save_all(ext_storage_path=ext_storage_path,
                           param_file_name=param_file_name,
                           net=net,
                           epoch = epoch,
                           fold=fold)
            if self.multi_label_lvl == 2:
                for i, rank in enumerate(['phylum', 'class', 'order', 'family', 'genus', 'species']):
                    list_train_score[rank].append(score_train[i])
                    list_val_score[rank].append(score_val[i])
                    list_train_acc[rank].append(acc_train[i])
                    list_val_acc[rank].append(acc_val[i])
                list_epochs.append(epoch)
            else:
                list_train_score.append(score_train)
                list_val_score.append(score_val)
                list_train_acc.append(acc_train)
                list_val_acc.append(acc_val)
                list_epochs.append(epoch)
        if self.multi_label_lvl == 2:
            csv_file_name = param_file_name.split('.')[0] + '_f' + str(fold) + '.csv'
            abs_path_csv_file_name = os.path.join(ext_storage_path, csv_file_name)
            dfpt = pd.DataFrame(list_train_score['phylum'])
            dfpv = pd.DataFrame(list_val_score['phylum'])
            dfct = pd.DataFrame(list_train_score['class'])
            dfcv = pd.DataFrame(list_val_score['class'])
            dfot = pd.DataFrame(list_train_score['order'])
            dfov = pd.DataFrame(list_val_score['order'])
            dfft = pd.DataFrame(list_train_score['family'])
            dffv = pd.DataFrame(list_val_score['family'])
            dfgt = pd.DataFrame(list_train_score['genus'])
            dfgv = pd.DataFrame(list_val_score['genus'])
            dfst = pd.DataFrame(list_train_score['species'])
            dfsv = pd.DataFrame(list_val_score['species'])
            dfe = pd.DataFrame(list_epochs)
            df = pd.concat([dfe,dfpt,dfpv,dfct,dfcv,dfot,dfov,dfft,dffv,dfgt,dfgv,dfst,dfsv], ignore_index=True, axis=1)
            df.columns = ['Epochs',
                          'Phylum_Train_PCC',
                          'Phylum_Test_PCC',
                          'Class_Train_PCC',
                          'Class_Test_PCC',
                          'Order_Train_PCC',
                          'Order_Test_PCC',
                          'Family_Train_PCC',
                          'Family_Test_PCC',
                          'Genus_Train_PCC',
                          'Genus_Test_PCC',
                          'Species_Train_PCC',
                          'Species_Test_PCC']
            df.to_csv(abs_path_csv_file_name)


            csv_file_name = param_file_name.split('.')[0] + '_f' + str(fold) + '_acc_' + '.csv'
            abs_path_csv_file_name = os.path.join(ext_storage_path, csv_file_name)
            dfpt = pd.DataFrame(list_train_acc['phylum'])
            dfpv = pd.DataFrame(list_val_acc['phylum'])
            dfct = pd.DataFrame(list_train_acc['class'])
            dfcv = pd.DataFrame(list_val_acc['class'])
            dfot = pd.DataFrame(list_train_acc['order'])
            dfov = pd.DataFrame(list_val_acc['order'])
            dfft = pd.DataFrame(list_train_acc['family'])
            dffv = pd.DataFrame(list_val_acc['family'])
            dfgt = pd.DataFrame(list_train_acc['genus'])
            dfgv = pd.DataFrame(list_val_acc['genus'])
            dfst = pd.DataFrame(list_train_acc['species'])
            dfsv = pd.DataFrame(list_val_acc['species'])
            dfe = pd.DataFrame(list_epochs)
            df = pd.concat([dfe,dfpt,dfpv,dfct,dfcv,dfot,dfov,dfft,dffv,dfgt,dfgv,dfst,dfsv], ignore_index=True, axis=1)
            df.columns = ['Epochs',
                          'Phylum_Train_ACC',
                          'Phylum_Test_ACC',
                          'Class_Train_ACC',
                          'Class_Test_ACC',
                          'Order_Train_ACC',
                          'Order_Test_ACC',
                          'Family_Train_ACC',
                          'Family_Test_ACC',
                          'Genus_Train_ACC',
                          'Genus_Test_ACC',
                          'Species_Train_ACC',
                          'Species_Test_ACC']
            df.to_csv(abs_path_csv_file_name)

        else:
            csv_file_name = param_file_name.split('.')[0]+'_f'+str(fold)+'.csv'
            abs_path_csv_file_name = os.path.join(ext_storage_path, csv_file_name)
            df = pd.DataFrame(list(zip(list_train_score, list_val_score, list_epochs)), columns=['scores_train', 'scores_test','epochs'])
            df.to_csv(abs_path_csv_file_name)

            csv_file_name = param_file_name.split('.')[0]+'_f'+str(fold)+'_acc_'+'.csv'
            abs_path_csv_file_name = os.path.join(ext_storage_path, csv_file_name)
            df = pd.DataFrame(list(zip(list_train_acc, list_val_acc, list_epochs)), columns=['acc_train', 'acc_test','epochs'])
            df.to_csv(abs_path_csv_file_name)
        return net

