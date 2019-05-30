Simple framework for hyperparameter optimization and cross-validation

## Installation
Pip installation for Python 3:

```pip3 install hopt```

## Basic usage of parameter randomization

Define your parameter search ranges:
```python
from hopt import Parameters, randoms

class HP(Parameters):
    step = randoms.Int(1, 20)
    batch_size = randoms.IntChoice([4, 8, 16, 128])
    dropout = randoms.Float(0.3, 0.7)
    lr = randoms.FloatExp(10, -1, -5)
    lr_decay = randoms.FloatChoice([0.1, 0.5, 0.95])
```

After calling randomize() the parameters get new values:

```python
HP.randomize()
print("Your random param values are: ", dict(HP.get_dict()))
```
```
Your randomized param values are:  {'batch_size': 16, 'dropout': 0.4720486083354445, 'lr': 0.023270515307809356, 'lr_decay': 0.95, 'step': 16}
```

Your params are just regular ints and floats, there's no need for calling any get() or value() method:

```python
current_iter = 10
if current_iter % HP.step:
    lr = HP.lr * HP.lr_decay
```
  
## Running hyperparam search with hopt

Hopt enables fast and simple execution of hyperparam search while keeping the code clean. It also performs most of the usual model saving and logging tasks.

First, define your parameter search ranges. You can also define static parameters. All params must be json serializable.

```python
from hopt import Parameters, randoms, search

class HP(Parameters):
    epochs = 10
    image_size = (512, 512)
    base_lr = randoms.FloatExp(10, -1, -5)
    lr_reduce_rate = randoms.FloatChoice([0.9, 0.95, 0.99])
    lr_reduce_patience = randoms.IntChoice([2, 5, 10]) 
```

Define a regular Keras train function. Pass Parameters and data as arguments. Append HoptCallback to fit() callbacks:

```python
def train(train_data, val_data, HP):
    # Callbacks
    lr_reduce = ReduceLROnPlateau(
        factor=HP.lr_reduce_rate, patience=HP.lr_reduce_patience,
        verbose=1, monitor='val_loss')
    hopt_callback = HoptCallback(
        metric_monitor='val_loss',
        metric_lower_better=False)
    callbacks = [lr_reduce, hopt_callback]
    
    # Model
    model = ...
    model.compile(optimizer=SGD(lr=HP.base_lr))
    
    # Fit
    model.fit_generator(
        generator=train_data,
        epochs=HP.epochs,
        callbacks=callbacks,
        validation_data=val_data)
```

Prepare data and run hyperparam search by using search() function:

```python
train_data = <some keras.utils.Sequence>
val_data = <some keras.utils.Sequence>

search(
    train_function=train,
    hyperparams=HP,
    out_dir='search-000',
    iterations=30,
    train_data=train_data,
    val_data=val_data)
```

hopt.search() will run your train() function multiple times (as defined by *iterations* argument). It will automatically save search results, best models and logs. It will also generate tensorboard events, including plots of best results for all tested hyperparam values, using all metrics available in logs.

Sample hyperparams plots:

![image](https://user-images.githubusercontent.com/30234642/58647452-e4e91100-8307-11e9-8720-3e6c244acec4.png)

**Note:** Due to tensorboard limitations, param values are encoded as tensorboard *steps*. Unfortunately *steps* must be integers, hence hopts encodes float values by multiplying by 10^6 and converting to int (see *base_lr* plot, on the right). Please keep that in mind while reading values from float param plots.
