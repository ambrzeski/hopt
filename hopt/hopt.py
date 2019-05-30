from __future__ import absolute_import
import types
from collections import OrderedDict
from multiprocessing import Process

import os
from .randoms import Param


# -------------------------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------------------------

class Parameters:
    """
    Parent class for holding hyperparameters, which can be defined as class attributes. Parameters should be json
    serializable. It can hold both regular parameters and search ranges defined in form of Params from randoms
    module (e.g. hopt.randoms.Int).

    Use by subclassing. Example:

    from hopt import Parameters, randoms

    class Hyperparams(Parameters):
        epochs = 10
        image_size = (512, 512)
        step = randoms.Int(1, 20)
        batch_size = randoms.IntChoice([4, 8, 16, 128])
        dropout = randoms.Float(0.3, 0.7)
        lr = randoms.FloatExp(10, -1, -5)
        lr_decay = randoms.FloatChoice([0.1, 0.5, 0.95])
    """

    # TODO: overload def __setattr__(self, name, value) to recreate random params fields in case
    # of direct value assignments e.g. MyParams.learning_rate = 0.5

    @classmethod
    def randomize(cls):
        for name, value in cls.get_params():
            setattr(cls, name, value.randomize())

    @classmethod
    def get_params(cls):
        return [(k, getattr(cls, k)) for k in dir(cls) if isinstance(getattr(cls, k), Param)]

    @classmethod
    def get_dict(cls):
        return OrderedDict(sorted([(k, getattr(cls, k)) for k in dir(cls)
                                   if not k.startswith('__') and not
                                   isinstance(getattr(cls, k), types.MethodType) and not
                                   isinstance(getattr(cls, k), types.FunctionType)]))

    @classmethod
    def set_param(cls, name, value):
        setattr(cls, name, value)

    @classmethod
    def update(cls, params):
        for k, v in params.get_dict().items():
            cls.set_param(k, v)

    def copy_class_to_instance(self):
        for attr, value in self.get_dict().items():
            setattr(self, attr, value)

    def copy_instance_to_class(self):
        d = OrderedDict(sorted([(k, getattr(self, k)) for k in dir(self)
                                if not k.startswith('__') and not
                                isinstance(getattr(self, k), types.MethodType) and not
                                isinstance(getattr(self, k), types.FunctionType)]))
        for attr, value in d.items():
            setattr(self.__class__, attr, value)

    @classmethod
    def serialize(cls):
        d = cls.get_dict()
        ranges = cls.get_ranges_dict()
        key = "search_ranges"
        while key in d:
            key = "_" + key
        d[key] = ranges
        return d

    @classmethod
    def get_ranges_dict(cls):
        ranges = OrderedDict()
        for k, param in cls.get_params():
            param_range_dict = OrderedDict()
            param_range_dict["type"] = param.__class__.__name__
            param_range_dict.update(param.__dict__)
            ranges[k] = param_range_dict
        return ranges


# -------------------------------------------------------------------------------------------------
# Search
# -------------------------------------------------------------------------------------------------

def search(train_function, hyperparams, out_dir, iterations, train_data=None, val_data=None, multiprocessing=False):
    """
    Initiates hyperparam search. Requires using hopt.keras.HoptCallback in Keras fit(). Before each iteration
    hyperparams are automatically randomized.

    :param train_function: function that performs training and validation which will be executed during search process
    :param hyperparams: hopt.Parameters class containing hyperparams
    :param out_dir: directory to save search results (models, logs)
    :param iterations: number of search iterations
    :param train_data: training data that will be passed to the train function
    :param val_data: validation data that will be passed to the train function
    :param multiprocessing: experimental, not safe to use
    """
    search_status = SearchStatus()
    search_status.out_dir = out_dir

    # Run search iterations
    for _ in range(iterations):
        __train(train_function, hyperparams, search_status, train_data, val_data, 0, multiprocessing)


def crossval_search(train_function, hyperparams, out_dir, iterations, max_epochs, data, multiprocessing=False):
    folds = len(data)
    print("Starting crossval search with {} folds".format(folds))

    # Checks
    if len(iterations) != len(max_epochs):
        raise ValueError("Iterations and max_epochs must have the same size.")

    search_status = SearchStatus()

    # Run search iterations
    for stage, iteration, max_epoch in zip(range(len(iterations)), iterations, max_epochs):
        for _ in range(iteration):
            for fold in range(folds):

                # Prepare data
                val_data = data[fold]
                train_data = data[:]
                train_data.pop(fold)

                search_status.out_dir = os.path.join(out_dir, "split{}".format(fold))

                __train(train_function, hyperparams, search_status, train_data, val_data, stage, multiprocessing)


def __train(train_function, hyperparams, search_status, train_data, val_data, stage, multiprocessing=False):
    # Randomize hyperparams
    hyperparams.randomize()

    # Cache search status
    search_status.stage = stage
    search_status.hyperparams = hyperparams()

    # Start training process
    if multiprocessing:
        search_status.hyperparams.copy_class_to_instance()
        p = Process(target=__process, args=(train_function, train_data, val_data, search_status))
        p.start()
        p.join()
    else:
        __process(train_function, train_data, val_data, search_status)


class SearchStatus(Parameters):
    out_dir = None
    stage = 0
    resume_model = None
    resume_id = None
    hyperparams = None


def __process(function, train_data, val_data, search_status):
    # Reload values to back to class attributes
    # (this seems mandatory in Python 3)
    search_status.hyperparams.copy_instance_to_class()
    search_status.copy_instance_to_class()

    # Run the function
    if train_data is None or val_data is None:
        function(search_status.hyperparams)
    else:
        function(train_data, val_data, search_status.hyperparams)
