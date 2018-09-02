from __future__ import absolute_import
import os
import glob
import json 
import datetime
import re

from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, TensorBoard

from .hopt import JsonEncoder, SearchStatus


LOG_DIRNAME = 'log'
HYPERPARAMS_DIRNAME = 'hyperparams'
MODELS_DIRNAME = 'models'


class HoptCallback(Callback):

    def __init__(self, test_generators=None, plot_hyperparams=False, workers=1):
        super(HoptCallback, self).__init__()

        self.test_generators = test_generators
        self.plot_hyperparams = plot_hyperparams
        self.workers = workers

        self.model = None
        self.params = None
        self.out_dir = None
        self.timestamp = None
        self.checkpoint_path = None
        self.log_path = None
        self.hyperparam_dir = None
        self.tensorboard_dir = None

        # Callbacks
        self.callbacks = None
        self.test_callback = None

    def initialize(self):

        print(SearchStatus.__dict__)
        self.out_dir = os.path.expanduser(SearchStatus.out_dir)

        # Prepare timestamp
        self.timestamp = get_timestamp()
        if SearchStatus.resume_model:
            self.timestamp = get_timestamp_from_path(SearchStatus.resume_model)
            if SearchStatus.resume_id is not None:
                self.timestamp += "_{}".format(SearchStatus.resume_id)

        # Prepare dirs
        log_dir = os.path.join(self.out_dir, LOG_DIRNAME)
        self.hyperparam_dir = os.path.join(self.out_dir, HYPERPARAMS_DIRNAME)
        models_dir = os.path.join(self.out_dir, MODELS_DIRNAME)
        self.tensorboard_dir = os.path.join(log_dir, self.timestamp)
        for d in [self.out_dir, log_dir, self.hyperparam_dir, models_dir, self.tensorboard_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Prepare paths
        self.checkpoint_path = models_dir + "/{val_loss:.4f}_{loss:.4f}_{epoch:02d}_" + self.timestamp + ".hdf5"
        self.log_path = os.path.join(log_dir, self.timestamp + ".csv")

    def on_train_begin(self, logs=None):
        self.initialize()
        saver = ModelCheckpoint(filepath=self.checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss')
        cleaner = Cleaner(self.out_dir, self.timestamp)
        logger = CSVLogger(filename=os.path.normpath(self.log_path), separator=';')
        hp_logger = HyperparamLogger(SearchStatus.hyperparams, self.hyperparam_dir, self.timestamp,
                                     self.plot_hyperparams)
        tensorboard = TensorBoard(self.tensorboard_dir)
        self.test_callback = TestEvaluator(self.test_generators)
        self.callbacks = [saver, cleaner, logger, hp_logger, tensorboard, self.test_callback]

        for c in self.callbacks:
            c.set_params(self.params)
            c.set_model(self.model)
            c.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for c in self.callbacks:
            c.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for c in self.callbacks:
            c.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        test_logs = self.test_callback.on_epoch_end(epoch, logs)
        full_logs = logs.copy()
        full_logs.update(test_logs)
        callbacks = self.callbacks[:]
        callbacks.remove(self.test_callback)
        for c in callbacks:
            c.on_epoch_end(epoch, full_logs)

    def on_batch_begin(self, batch, logs=None):
        for c in self.callbacks:
            c.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for c in self.callbacks:
            c.on_batch_end(batch, logs)

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model


class Cleaner(Callback):
    def __init__(self, out_dir, timestamp):
        super(Cleaner, self).__init__()
        self.out_dir = out_dir
        self.timestamp = timestamp

    def on_epoch_end(self, batch, logs=()):
        saved_models = sorted(glob.glob(os.path.join(self.out_dir, MODELS_DIRNAME, "*{}.hdf5".format(self.timestamp))))

        # Leave only best model, delete the rest
        for path in saved_models[1:]:
            print("Removing model: ", path)
            os.remove(path)


class HyperparamLogger(Callback):
    def __init__(self, hyperparams, hyperparam_dir, timestamp, plot):
        super(HyperparamLogger, self).__init__()
        self.hyperparams = hyperparams
        self.hyperparam_dir = hyperparam_dir
        self.timestamp = timestamp
        self.plot = plot
        self.file_path = os.path.join(hyperparam_dir, timestamp + ".json")
        self.top_val_loss = float("inf")
        self.top_val_loss_epoch = None

    def on_train_begin(self, logs=None):
        # Dump hyperparams
        d = self.hyperparams.get_dict()
        with open(self.file_path, 'w') as f:
            json.dump(d, f, indent=4, cls=JsonEncoder)

    def on_train_end(self, logs=None):
        # Dump hyperparams with loss and acc under new name
        d = self.hyperparams.get_dict()
        d['val_loss'] = self.top_val_loss
        d['epoch'] = self.top_val_loss_epoch
        parentdir = os.path.dirname(self.file_path)
        new_path = os.path.join(parentdir, "{:.4f}_{:02d}_{}.json").format(self.top_val_loss,
                                                                           self.top_val_loss_epoch, self.timestamp)
        with open(new_path, 'w') as f:
            json.dump(d, f, indent=4, cls=JsonEncoder)

        # Remove old dumped hyperparams
        os.remove(self.file_path)

        # Plot hyperparams
        if self.plot:
            plot_hyperparams(parentdir)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss < self.top_val_loss:
            self.top_val_loss = val_loss
            self.top_val_loss_epoch = epoch


class EarlyStopHugeLoss(Callback):

    def __init__(self, min_loss):
        super(EarlyStopHugeLoss, self).__init__()
        self.min_loss = min_loss

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') > self.min_loss:
            self.model.stop_training = True


class TestEvaluator(Callback):

    def __init__(self, generators, workers=1):
        super(TestEvaluator, self).__init__()
        self.generators = generators
        self.workers = workers

    def on_epoch_end(self, epoch, logs=None):
        if self.generators is None:
            return
        results = {}
        for i, generator in enumerate(self.generators):
            if generator is None:
                continue
            suffix = '' if len(self.generators) == 1 else str(i+1)
            print('Evaluating on test{} set...'.format(suffix))
            metrics = self.model.evaluate_generator(generator, workers=self.workers)
            metrics_names = ['test' + suffix + '_' + m for m in self.model.metrics_names]
            results.update(dict(zip(metrics_names, metrics)))
        if results:
            print(results)
        return results


# PARAMS = ['base_lr', 'dropout', 'batch_size', 'momentum', 'lr_decay', 'l2_penalty']
PARAMS = ['base_lr', 'lr_reduce_rate']
METRICS = ['val_loss']
PARAM_SCALE_HINTS = {'base_lr': 'log', 'lr_decay': 'log', 'l2_penalty': 'log'}
METRIC_SCALE_HINTS = {}
MAX_LOSS = 1.0


def plot_hyperparams(query):
    paths = glob.glob(query)
    for path in paths:

        # Read all dumped hyperparams
        json_paths = glob.glob(os.path.join(path, "*_*_*.json"))  # *_* <- hack for skipping not ready (not renamed) dicts
        dicts = []
        for jp in json_paths:
            with open(jp) as f:
                dicts.append(json.load(f))

        # Plot
        for m in METRICS:
            for p in PARAMS:
                save_path = os.path.join(path, "{}_{}.png".format(m, p))
                plot_param(p, m, dicts, save_path)


def plot_param(name, metric, dicts, save_path):
    import matplotlib.pyplot as plt
    param_values = [x[name] for x in dicts]
    m = [x[metric] for x in dicts]
    try:
        plt.clf()
        plt.plot(param_values, m, 'ro')
        plt.xlabel(name)
        plt.ylabel(metric)

        if metric == 'val_loss' and max(m) > MAX_LOSS:
            plt.ylim([0, MAX_LOSS])

        if metric in METRIC_SCALE_HINTS:
            plt.yscale(METRIC_SCALE_HINTS[metric])

        if name in PARAM_SCALE_HINTS:
            plt.xscale(PARAM_SCALE_HINTS[name])

        plt.savefig(save_path)
        plt.clf()
    except Exception as e:
        plt.clf()
        print("Error plotting hyperparams: ", e)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_timestamp_from_path(path):
    return re.split("[_.]", path)[-2]
