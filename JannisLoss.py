from mxnet.gluon.loss import Loss

import numpy as np
from mxnet import ndarray
from mxnet.base import numeric_types
from mxnet.block import HybridBlock
from mxnet.util import is_np_array

def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    if F is ndarray:
        return x.reshape(y.shape)
    elif is_np_array():
        F = F.npx
    return F.reshape_like(x, y)

def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        if is_np_array():
            loss = loss * sample_weight
        else:
            loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss

class JannisLoss(Loss):


    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(JannisLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        # get a function handle
        if is_np_array():
            log_softmax = F.npx.log_softmax
            pick = F.npx.pick
        else:
            log_softmax = F.log_softmax
            pick = F.pick
        # get log softmax
        if not self._from_logits:
            pred = log_softmax(pred, self._axis)
        #
        if self._sparse_label:
            loss = -pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -(pred * label).sum(axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)

        if is_np_array():
            if F is ndarray:
                return loss.mean(axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return loss.mean(axis=self._batch_axis, exclude=True)