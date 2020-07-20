from mxnet.gluon import loss as gloss
from mxnet.gluon.loss import _apply_weighting, _reshape_like

class CategoricalLoss(gloss.Loss):
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(CategoricalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class FocalLoss(gloss.Loss):
    pass
    # def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
    #              batch_axis=0, **kwargs):
    #     super(CategoricalLoss, self).__init__(weight, batch_axis, **kwargs)
    #     self._axis = axis
    #     self._sparse_label = sparse_label
    #     self._from_logits = from_logits
    #
    # def hybrid_forward(self, F, pred, label, sample_weight=None):
    #     if not self._from_logits:
    #         pred = F.log_softmax(pred, self._axis)
    #     if self._sparse_label:
    #         loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
    #     else:
    #         label = _reshape_like(F, label, pred)
    #         loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
    #     loss = _apply_weighting(F, loss, self._weight, sample_weight)
    #     return F.mean(loss, axis=self._batch_axis, exclude=True)
