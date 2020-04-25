from mxnet.gluon.loss import Loss
from mxnet.gluon.loss import _apply_weighting
from mxnet.gluon.loss import _reshape_like


class JannisLoss(Loss):


    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(JannisLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss_0 = F.pick(pred, label[:, 0], axis=self._axis, keepdims=True)
            loss_1 = F.pick(pred, label[:, 1], axis=self._axis, keepdims=True)
            loss_2 = F.pick(pred, label[:, 2], axis=self._axis, keepdims=True)
            loss_3 = F.pick(pred, label[:, 3], axis=self._axis, keepdims=True)
            loss_4 = F.pick(pred, label[:, 4], axis=self._axis, keepdims=True)
            loss_5 = F.pick(pred, label[:, 5], axis=self._axis, keepdims=True)
            loss = -1*(loss_0+loss_1+loss_2+loss_3+loss_4+loss_5)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred * label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
