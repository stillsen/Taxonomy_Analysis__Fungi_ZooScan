
"""Online evaluation metric module."""
from __future__ import absolute_import
import math


import numpy

# from .base import numeric_types, string_types
# from . import ndarray
from mxnet import ndarray
# from . import registry


def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction

    Parameters
    ----------
    labels : list of `NDArray`
        The labels of the data.

    preds : list of `NDArray`
        Predicted values.

    wrap : boolean
        If True, wrap labels/preds in a list if they are single NDArray

    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

    if wrap:
        if isinstance(labels, ndarray.ndarray.NDArray):
            labels = [labels]
        if isinstance(preds, ndarray.ndarray.NDArray):
            preds = [preds]

    return labels, preds

class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> NDArray
            name to array mapping for labels.

        preds : OrderedDict of str -> NDArray
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))



class _BinaryClassificationMetrics(object):
    """
    Private container class for classification metric statistics. True/false positive and
     true/false negative counts are sufficient statistics for various classification metrics.
    This class provides the machinery to track those statistics across mini-batches of
    (label, prediction) pairs.
    """

    def __init__(self):
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.true_negatives = 0

    def update_binary_stats(self, label, pred):
        """
        Update various binary classification counts for a single (label, pred)
        pair.

        Parameters
        ----------
        label : `NDArray`
            The labels of the data.

        pred : `NDArray`
            Predicted values.
        """
        pred = pred.asnumpy()
        label = label.asnumpy().astype('int32')
        pred_label = numpy.argmax(pred, axis=1)

        check_label_shapes(label, pred)
        if len(numpy.unique(label)) > 2:
            raise ValueError("%s currently only supports binary classification."
                             % self.__class__.__name__)
        pred_true = (pred_label == 1)
        pred_false = 1 - pred_true
        label_true = (label == 1)
        label_false = 1 - label_true

        self.true_positives += (pred_true * label_true).sum()
        self.false_positives += (pred_true * label_false).sum()
        self.false_negatives += (pred_false * label_true).sum()
        self.true_negatives += (pred_false * label_false).sum()

    @property
    def precision(self):
        if self.true_positives + self.false_positives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_positives)
        else:
            return 0.

    @property
    def recall(self):
        if self.true_positives + self.false_negatives > 0:
            return float(self.true_positives) / (self.true_positives + self.false_negatives)
        else:
            return 0.

    @property
    def fscore(self):
        if self.precision + self.recall > 0:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            return 0.

    @property
    def matthewscc(self):
        """
        Calculate the Matthew's Correlation Coefficent
        """
        if not self.total_examples:
            return 0.

        true_pos = float(self.true_positives)
        false_pos = float(self.false_positives)
        false_neg = float(self.false_negatives)
        true_neg = float(self.true_negatives)
        terms = [(true_pos + false_pos),
                 (true_pos + false_neg),
                 (true_neg + false_pos),
                 (true_neg + false_neg)]
        denom = 1.
        for t in filter(lambda t: t != 0., terms):
            denom *= t
        return ((true_pos * true_neg) - (false_pos * false_neg)) / math.sqrt(denom)

    @property
    def total_examples(self):
        return self.false_negatives + self.false_positives + \
               self.true_negatives + self.true_positives

    def reset_stats(self):
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0



class F1(EvalMetric):
    """Computes the F1 score of a binary classification problem.

    The F1 score is equivalent to harmonic mean of the precision and recall,
    where the best value is 1.0 and the worst value is 0.0. The formula for F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    The formula for precision and recall is::

        precision = true_positives / (true_positives + false_positives)
        recall    = true_positives / (true_positives + false_negatives)

    .. note::

        This F1 score only supports binary classification.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    average : str, default 'macro'
        Strategy to be used for aggregating across mini-batches.
            "macro": average the F1 scores for each batch.
            "micro": compute a single F1 score across all batches.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0., 1., 1.])]
    >>> f1 = mx.metric.F1()
    >>> f1.update(preds = predicts, labels = labels)
    >>> print f1.get()
    ('f1', 0.8)
    """

    def __init__(self, name='f1',
                 output_names=None, label_names=None, average="macro"):
        self.average = average
        self.metrics = _BinaryClassificationMetrics()
        EvalMetric.__init__(self, name=name,
                            output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            self.metrics.update_binary_stats(label, pred)

        if self.average == "macro":
            self.sum_metric += self.metrics.fscore
            self.num_inst += 1
            self.metrics.reset_stats()
        else:
            self.sum_metric = self.metrics.fscore * self.metrics.total_examples
            self.num_inst = self.metrics.total_examples

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.sum_metric = 0.
        self.num_inst = 0.
        self.metrics.reset_stats()

