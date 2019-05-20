import os
import glob
import json


# PARAMS = ['base_lr', 'dropout', 'batch_size', 'momentum', 'lr_decay', 'l2_penalty']
PARAMS = ['base_lr', 'lr_reduce_rate']
METRICS = ['val_loss']
PARAM_SCALE_HINTS = {'base_lr': 'log', 'lr_decay': 'log', 'l2_penalty': 'log'}
METRIC_SCALE_HINTS = {}
MAX_LOSS = 1.0


def create_dirs(paths_list):
    for e in paths_list:
        os.makedirs(e, exist_ok=True)


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
