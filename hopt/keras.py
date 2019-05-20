from __future__ import absolute_import
import os
import glob
import json 
import datetime
import re

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, TensorBoard, K

from .hopt import JsonEncoder, SearchStatus
from .utils import plot_hyperparams, create_dirs


LOG_DIRNAME = 'log'
HYPERPARAMS_DIRNAME = 'hyperparams'
RESULTS_DIRNAME = 'results'
MODELS_DIRNAME = 'models'


class HoptCallback(Callback):

    def __init__(self, model_prefix='{val_loss:.4f}_{epoch:02d}', model_lower_better=True, model_monitor='val_loss',
                 keep_models=1, save_tf_graphs=False, test_generators=None, workers=1, plot_hyperparams=False):
        """
        Callback for hyper parameter search.

        :param model_prefix: pattern for model filename prefix, default is: '{val_loss:.4f}_{epoch:02d}'
        :param model_lower_better: True if for alphabetically sorted models first (lower is the best), False otherwise
        :param model_monitor: metric to monitor for saving best model
        :param keep_models: how many models to keep during each iterations; 0 keeps all models
        :param save_tf_graphs: save TensorFlow frozen graphs along with Keras models
        :param test_generators: additional Keras Sequences to be evaluated during training as test sets
        :param workers: workers to use for evaluating on test_generators
        :param plot_hyperparams: should hyperparams be plotted? (requires matplotlib)
        """
        super(HoptCallback, self).__init__()

        self.model_prefix = model_prefix
        self.model_lower_better = model_lower_better
        self.model_monitor = model_monitor
        self.keep_models = keep_models
        self.save_tf_graphs = save_tf_graphs
        self.test_generators = test_generators
        self.plot_hp = plot_hyperparams
        self.workers = workers

        self.model = None
        self.params = None
        self.out_dir = None
        self.timestamp = None
        self.checkpoint_path = None
        self.log_path = None
        self.hyperparam_dir = None
        self.models_dir = None
        self.results_dir = None
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
        self.models_dir = os.path.join(self.out_dir, MODELS_DIRNAME)
        self.results_dir = os.path.join(self.out_dir, RESULTS_DIRNAME)
        self.tensorboard_dir = os.path.join(log_dir, self.timestamp)
        dirs = [self.out_dir, log_dir, self.hyperparam_dir, self.models_dir, self.tensorboard_dir, self.results_dir]
        create_dirs(dirs)

        # Prepare paths
        self.checkpoint_path = "{}/{}_{}.hdf5".format(self.models_dir, self.model_prefix, self.timestamp)
        self.log_path = os.path.join(log_dir, self.timestamp + ".csv")

    def on_train_begin(self, logs=None):
        self.initialize()
        self.callbacks = self.prepare_callbacks()

        for c in self.callbacks:
            c.set_params(self.params)
            c.set_model(self.model)
            c.on_train_begin(logs)

    def prepare_callbacks(self):
        """
        Prepares callbacks for HoptCallback
        """
        callbacks = []

        # Saver callback
        saver_callback_cls = MultiGraphSaver if self.save_tf_graphs else ModelCheckpoint
        saver = saver_callback_cls(
            filepath=self.checkpoint_path,
            save_best_only=True,
            verbose=1,
            monitor=self.model_monitor)
        callbacks.append(saver)

        # Cleaner callback - must be added after saver
        cleaner = Cleaner(
            out_dir=self.out_dir,
            timestamp=self.timestamp,
            model_lower_better=self.model_lower_better,
            keep_models=self.keep_models)
        callbacks.append(cleaner)

        # Logger callback
        logger = CSVLogger(
            filename=os.path.normpath(self.log_path),
            separator=';')
        callbacks.append(logger)

        # Hyper param logger callback
        hp_logger = HyperparamLogger(
            hyperparams=SearchStatus.hyperparams,
            hyperparam_dir=self.hyperparam_dir,
            timestamp=self.timestamp,
            plot=self.plot_hp)
        callbacks.append(hp_logger)

        # Best result logger
        result_logger = BestResultLogger(
            monitored_metric=self.model_monitor,
            results_dir=self.results_dir,
            timestamp=self.timestamp)
        callbacks.append(result_logger)

        # Tensorboard callback
        tensorboard = TensorBoard(log_dir=self.tensorboard_dir)
        callbacks.append(tensorboard)

        # Test evaluator callback
        self.test_callback = TestEvaluator(generators=self.test_generators)
        callbacks.append(self.test_callback)

        return callbacks

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
    def __init__(self, out_dir, timestamp, model_lower_better, keep_models):
        super(Cleaner, self).__init__()
        self.out_dir = out_dir
        self.timestamp = timestamp
        self.model_lower_better = model_lower_better
        self.keep_models = keep_models

    def on_epoch_end(self, batch, logs=()):
        if self.keep_models > 0:
            exts = ["hdf5", "pb"]
            queries = [os.path.join(self.out_dir, MODELS_DIRNAME, "*{}.{}".format(self.timestamp, ext)) for ext in exts]
            for query in queries:
                # Find models
                saved_models = sorted(glob.glob(query))
                if self.model_lower_better:
                    to_remove = saved_models[self.keep_models:]
                else:
                    to_remove = saved_models[:-self.keep_models]

                # Leave only best model, delete the rest
                for path in to_remove:
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

    def on_train_begin(self, logs=None):
        # Dump hyperparams
        d = self.hyperparams.get_dict()
        with open(self.file_path, 'w') as f:
            json.dump(d, f, indent=4, cls=JsonEncoder)

    def on_train_end(self, logs=None):
        # Plot hyperparams
        parent = os.path.dirname(self.file_path)
        if self.plot:
            plot_hyperparams(parent)


class BestResultLogger(Callback):
    def __init__(self, monitored_metric, results_dir, timestamp):
        """
        Saves results of the best model selected by the specified metric. Results are saved in json file.

        :param monitored_metric: metric used to select the best model
        :param results_dir: directory path to save the results
        :param timestamp: timestamp of the training iteration, will be used as filename
        """
        super().__init__()
        self.monitored_metric = monitored_metric
        self.results_dir = results_dir
        self.timestamp = timestamp
        self.file_path = os.path.join(results_dir, timestamp + ".json")
        self.best = None
        self.best_val = None

    def on_epoch_end(self, batch, logs=()):
        if not self.best_val or logs[self.monitored_metric] >= self.best_val:
            self.best = dict(logs)
            print("Best result: ", self.best)
            with open(self.file_path, 'w') as f:
                json.dump(self.best, f, indent=4, cls=self.NumpyEncoder)

    class NumpyEncoder(json.JSONEncoder):
        """ Numpy types encoder """
        def default(self, obj):
            float_types = (np.float_, np.float16, np.float32, np.float64)
            int_types = (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
            if isinstance(obj, int_types):
                return int(obj)
            elif isinstance(obj, float_types):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)


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
        if isinstance(generators, list):
            self.generators = generators
        else:
            self.generators = [generators]
        self.workers = workers

    def on_epoch_end(self, epoch, logs=None):
        if not self.generators:
            return {}

        metrics = []
        metrics_names = []
        for i, generator in enumerate(self.generators):
            if generator is None or not len(generator):
                continue

            # Prepare test set name suffix
            suffix = ''
            if len(self.generators) > 1:
                suffix += str(i+1)
                if hasattr(generator, 'name') and generator.name:
                    suffix += '_' + generator.name

            print('Evaluating on test{} set...'.format(suffix))
            metrics.append(self.model.evaluate_generator(generator, workers=self.workers))
            metrics_names.append(['test' + suffix + '_' + m for m in self.model.metrics_names])

        # Calculate mean metrics for all test sets
        if len(self.generators) > 1 and len(metrics):
            mean_metrics = np.mean(metrics, axis=0)
            metrics.append(mean_metrics)
            metrics_names.append(['test_' + m for m in self.model.metrics_names])

        # Prepare the result
        results = {}
        for names, m in zip(metrics_names, metrics):
            results.update(dict(zip(names, m)))
        if results:
            print(results)

        return results


class MultiGraphSaver(ModelCheckpoint):

    def on_train_begin(self, logs=None):

        class TFModelWrapper:
            def __init__(self, model):
                self.model = model

            def save(self, save_path, overwrite):
                # Save Keras model
                self.model.save(save_path, overwrite)

                # Save TF graph
                frozen_graph = self.freeze_session(K.get_session(), output_names=[self.model.output.op.name])
                if save_path.endswith(".hdf5") or save_path.endswith(".h5"):
                    save_path = save_path[:save_path.rfind('.')]
                save_path += ".pb"
                dirpath, filename = os.path.split(save_path)
                tf.train.write_graph(frozen_graph, dirpath, filename, as_text=False)

            @staticmethod
            def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
                """
                Freezes the state of a session into a prunned computation graph.

                Creates a new computation graph where variable nodes are replaced by
                constants taking their current value in the session. The new graph will be
                prunned so subgraphs that are not neccesary to compute the requested
                outputs are removed.
                @param session The TensorFlow session to be frozen.
                @param keep_var_names A list of variable names that should not be frozen,
                                      or None to freeze all the variables in the graph.
                @param output_names Names of the relevant graph outputs.
                @param clear_devices Remove the device directives from the graph for better portability.
                @return The frozen graph definition.
                """
                from tensorflow.python.framework.graph_util import convert_variables_to_constants
                graph = session.graph
                with graph.as_default():
                    freeze_var_names = list(
                        set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
                    output_names = output_names or []
                    output_names += [v.op.name for v in tf.global_variables()]
                    input_graph_def = graph.as_graph_def()
                    if clear_devices:
                        for node in input_graph_def.node:
                            node.device = ""
                    frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                                  output_names, freeze_var_names)
                    return frozen_graph

        super().on_train_begin()
        self.model = TFModelWrapper(self.model)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_timestamp_from_path(path):
    return re.split("[_.]", path)[-2]
