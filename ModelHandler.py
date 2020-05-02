import os, time
import mxnet as mx
from mxnet import gluon, init, gpu
from mxnet import autograd as ag
from gluoncv.model_zoo import get_model
import pandas as pd
from JannisLoss import JannisLoss

from mxnet.ndarray.contrib import isnan


class ModelHandler:
    def __init__(self,
                 classes,
                 # epochs = 10,
                 metrics,
                 rank=None,
                 learning_rate = 0.001,
                 batch_size = 1,
                 momentum = 0.9,
                 wd=0.0001,
                 num_gpus=1,
                 num_workers=1,
                 model_name = 'densenet169',
                 pretrained = True,
                 multi_label_lvl = 1,
                 ):
        # Multi-label lvl 1: Binary Relevance Approach
        # Multi-label lvl 2:  Multi-Label/Multi/Class with sigmoid activation function and binary cross entropy loss

        # self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum
        self.wd = wd
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.metrics = metrics

        self.gpus = mx.test_utils.list_gpus()
        self.ctx = [mx.gpu(i) for i in self.gpus] if len(self.gpus) > 0 else [mx.cpu()]
        # self.ctx = gpu()

        self.multi_label_lvl = multi_label_lvl
        self.rank = rank

        self.net = self.setup_net(multi_label_lvl=multi_label_lvl,
                                  model_name=model_name,
                                  pretrained=pretrained,
                                  classes=classes,
                                  ctx=self.ctx,
                                  rank=rank)

    def setup_net(self, multi_label_lvl, model_name, pretrained, classes, ctx, rank):
        # Multi-label lvl 1: Binary Relevance Approach
        # Multi-label lvl 2:  Multi-Label/Multi/Class with sigmoid activation function and binary cross entropy loss
        finetune_net = None
        if pretrained:
            if multi_label_lvl == 1:
                pretrained_net = get_model(model_name, pretrained=True)
                # pretrained_net = get_model(model_name, classes=classes, pretrained=True)
                finetune_net = get_model(model_name, classes=classes)

                finetune_net.features = pretrained_net.features
                finetune_net.output.initialize(init.Xavier(), ctx=ctx)
                # The model parameters in output will be updated using a learning rate ten
                # times greater
                finetune_net.output.collect_params().setattr('lr_mult', 10)

                finetune_net.collect_params().reset_ctx(ctx)
                finetune_net.hybridize()

                self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

            if multi_label_lvl == 2: # rn the same ml2 uses the same  loss and activation as ml1, bc classes not independent
                pretrained_net = get_model(model_name, pretrained=True)
                finetune_net = get_model(model_name, classes=classes)

                finetune_net.features = pretrained_net.features
                # finetune_net.output = nn.Dense(classes, activation='sigmoid')
                finetune_net.output.initialize(init.Xavier(), ctx=ctx)
                # The model parameters in output will be updated using a learning rate ten
                # times greater
                finetune_net.output.collect_params().setattr('lr_mult', 10)

                finetune_net.collect_params().reset_ctx(ctx)
                finetune_net.hybridize()

                # self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
                self.loss_fn = JannisLoss(rank)

        return finetune_net

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
            if self.multi_label_lvl == 2:
                label = gluon.utils.split_and_load(batch.label[0][:,self.rank], ctx_list=ctx, batch_axis=0, even_split=False)
            else:
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        val_data.reset()
        return metric.get()

    def _save_all(self, ext_storage_path, param_file_name, net, epoch, fold):
        print('\t\t\tsaving parameters')
        e_param_file_name = param_file_name.split('.')[0]+'_e'+str(epoch)+'_f'+str(fold)+'.param'
        abs_path_param_file_name = os.path.join(ext_storage_path, e_param_file_name)
        net.save_parameters(abs_path_param_file_name)



    def train(self, train_iter, val_iter, epochs, param_file_name, fold, ext_storage_path=''):
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

        print('\tComputing initial validation score...')
        val_names, score_val = self.evaluate(net, val_iter, ctx, metric=self.metrics)
        print('\t[Initial] validation: %s'%(self.metric_str(val_names, score_val)))
        list_train_score = []
        list_val_score = []
        list_epochs = []
        for epoch in range(epochs):
            tic = time.time()
            btic = time.time()
            metric.reset()

            for batch in train_iter:
                # print('training... epoch:  %d  batch_no:  %s' %(epoch,i))
                # the model zoo models expect normalized images
                # data = gluon.utils.split_and_load(batch.data, ctx_list=ctx, batch_axis=0, even_split=False)
                # label = gluon.utils.split_and_load(batch.label, ctx_list=ctx, batch_axis=0, even_split=False)
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
                with ag.record():
                    outputs = [net(X) for X in data]
                    loss=[]
                    for yhat, y in zip(outputs, label):
                        loss = [*loss, loss_fn(yhat, y)]
                        if isnan(loss):#any(isnan(loss))
                            print('nan in loss')
                            print(loss)
                            print(yhat)
                            print(y)
                    for l in loss:
                        l.backward()
                trainer.step(batch_size)
                metric.update(label, outputs)

                # if log_interval and not (i+1)%log_interval:
                #     name, score_train = metric.get()
                #     # print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                #     #                epoch, i, batch_size/(time.time()-btic), metric_str(name, value)))
                # btic = time.time()

            name, score_train = metric.get()
            metric.reset()
            train_iter.reset()
            print('\t[Fold %d Epoch %d] training: %s'%(fold, epoch, self.metric_str(name, score_train)))
            print('\t[Fold %d Epoch %d] time cost: %f'%(fold, epoch, time.time()-tic))
            val_names, score_val = self.evaluate(net, val_iter, ctx, self.metrics)
            print('\t[Fold %d Epoch %d] validation: %s'%(fold, epoch, self.metric_str(val_names, score_val)))
            # train_loss /= num_batch

            # ext_storage_path, param_file_name, app_file_name, net_list, score_list, app
            self._save_all(ext_storage_path=ext_storage_path,
                           param_file_name=param_file_name,
                           net=net,
                           epoch = epoch,
                           fold=fold)
            list_train_score.append(score_train)
            list_val_score.append(score_val)
            list_epochs.append(epoch)

        csv_file_name = param_file_name.split('.')[0]+'_f'+str(fold)+'.csv'
        abs_path_csv_file_name = os.path.join(ext_storage_path, csv_file_name)
        df = pd.DataFrame(list(zip(list_train_score, list_val_score, list_epochs)), columns=['scores_train', 'scores_test','epochs'])
        df.to_csv(abs_path_csv_file_name)

        return net
