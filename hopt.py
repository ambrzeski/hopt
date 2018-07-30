from __future__ import absolute_import
import types
import json
from collections import OrderedDict
from multiprocessing import Process

import os
from .randoms import Param


# -------------------------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------------------------

class Parameters(object):
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
                                   isinstance(getattr(cls, k), types.MethodType)]))

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
                                   isinstance(getattr(self, k), types.MethodType)]))
        for attr, value in d.items():
            setattr(self.__class__, attr, value)


class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        # if isinstance(obj, aug.MixedAugmenter):
        #     return obj.augmenters
        # if isinstance(obj, aug.Augmenter):
        #     return obj.name
        return json.JSONEncoder.default(self, obj)


# -------------------------------------------------------------------------------------------------
# Search
# -------------------------------------------------------------------------------------------------

def search(train_function, hyperparams, out_dir, iterations, max_epochs, train_data, val_data, multiprocessing=False):
    # Checks
    if len(iterations) != len(max_epochs):
        raise ValueError("Iterations and max_epochs must have the same size.")

    search_status = SearchStatus()
    search_status.out_dir = out_dir

    # Run search iterations
    for stage, iteration, max_epoch in zip(range(len(iterations)), iterations, max_epochs):
        for _ in range(iteration):
            train(train_function, hyperparams, search_status, train_data, val_data, stage, multiprocessing)


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

                train(train_function, hyperparams, search_status, train_data, val_data, stage, multiprocessing)


def train(train_function, hyperparams, search_status, train_data, val_data, stage, multiprocessing=False):
    # Randomize hyperparams
    hyperparams.randomize()

    if stage > 0:
        # TODO: support multistage
        pass

    # Cache search status
    search_status.stage = stage
    search_status.hyperparams = hyperparams()
    search_status.hyperparams.copy_class_to_instance()

    # Start training process
    if multiprocessing:
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
    function(train_data, val_data, search_status.hyperparams)
