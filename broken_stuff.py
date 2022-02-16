#!/usr/bin/env python
# coding: utf-8

# In[1]:


import deepchem as dc
import tensorflow as tf
import sklearn as sk
import numpy as np
import pandas as pd

print("TensorFlow version: " + tf.__version__)
print("DeepChe, version: " + dc.__version__)

metric_labels=['mean_squared_error','pearson_r2_score',
               'mae_score', 'rmse']


metric1 = dc.metrics.Metric(dc.metrics.mean_squared_error)
metric2 = dc.metrics.Metric(dc.metrics.pearson_r2_score)
metric3 = dc.metrics.Metric(dc.metrics.mae_score)
metrics = [metric1, metric2, metric3]
metric_selector = 2 #which metric to use for callback


# In[2]:


def get_them_metrics(
        model,
        datasets,
        metrics,
        metric_labels,
        transformers=[],
):
    """calculates metrics for a run
    model: trained model
    # datasets: tuple of datasets
    # metrics: list of metric objects
    # metric labels: sensible labels"""
    out = []
    for dataset in datasets:
        if transformers == []:
            egg = model.evaluate(
                dataset,
                metrics)
        else:
            egg = model.evaluate(
                dataset,
                metrics,
                transformers=transformers)
        for metric_label in metric_labels:
            if metric_label == 'rmse':
                out.append(np.sqrt(egg['mean_squared_error']))
            else:
                out.append(egg[metric_label])
    return out


# ## Multitask regressor

# In[3]:


patience=3

tasks, datasets, transformers = dc.molnet.load_qm7(
    shard_size=2000,
    featurizer=dc.feat.CoulombMatrix
    (max_atoms=23),
    splitter='stratified',
    move_mean=False)

callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=3)

# the datasets object is already split into the train, validation and test dataset 
train_dataset, valid_dataset, test_dataset = datasets

fit_transformers = [dc.trans.CoulombFitTransformer(train_dataset)]

# this loads in a general purpose regression model
model = dc.models.MultitaskFitTransformRegressor(
    n_tasks = len(test_dataset.tasks), # size of y, we have one output task here: finding toxicity
    n_features = [23,23],
    fit_transformers = fit_transformers # number of input features, i.e. the length of the ECFPs
)

# this sets up a callback on the validation
callback = dc.models.ValidationCallback(
            valid_dataset,
            patience,
            metrics[metric_selector])
# fit da model
model.fit(train_dataset, nb_epoch=100, callbacks=callback)


# In[4]:


# little function to calc metrics on this data
out=get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers)
# makes a nice dataframe
pd_out = pd.DataFrame([out], columns=['tr_mse', 'tr_r2', 'tr_mae', 'tr_rmse',
                                        'val_mse', 'val_r2', 'val_mae', 'val_rmse',
                                        'te_mse', 'te_r2', 'te_mae', 'te_rmse'])
print(pd_out)


# ## DTNN

# In[5]:


patience=3

# This loads the data without shuffling or splitting
tasks, datasets, transformers = dc.molnet.load_qm7(
    shard_size=2000,
    featurizer=dc.feat.CoulombMatrix(max_atoms=23),
    splitter='stratified')

callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=3)

# the datasets object is already split into the train, validation and test dataset 
train_dataset, valid_dataset, test_dataset = datasets

fit_transformers = [dc.trans.CoulombFitTransformer(train_dataset)]

# this loads in a general purpose regression model
model = dc.models.DTNNModel(
    n_tasks = len(test_dataset.tasks) # number of input features, i.e. the length of the ECFPs
)

# this sets up a callback on the validation
callback = dc.models.ValidationCallback(
            valid_dataset,
            patience,
            metrics[metric_selector])
# fit da model
model.fit(train_dataset, nb_epoch=100, callbacks=callback)


# In[6]:


# little function to calc metrics on this data
out=get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers)
# makes a nice dataframe
pd_out = pd.DataFrame([out], columns=['tr_mse', 'tr_r2', 'tr_mae', 'tr_rmse',
                                        'val_mse', 'val_r2', 'val_mae', 'val_rmse',
                                        'te_mse', 'te_r2', 'te_mae', 'te_rmse'])
print(pd_out)


# ## Kernel ridge regression

# In[7]:


train_dataset_X = [x.flatten() for x in train_dataset.X[0:5]]
train_dataset_y = [x.flatten() for x in train_dataset.y[0:5]]


# In[8]:


from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error

tasks, datasets, transformers = dc.molnet.load_qm7(
    featurizer=dc.feat.CoulombMatrix(max_atoms=23), 
    splitter='stratified', 
    move_mean=False)

train_dataset, valid_dataset, test_dataset = datasets

train_dataset, valid_dataset, test_dataset = datasets

train_dataset_X = [x.flatten() for x in train_dataset.X]
train_dataset_y = [x.flatten() for x in train_dataset.y]

test_dataset_X = [x.flatten() for x in test_dataset.X]
test_dataset_y = [x.flatten() for x in test_dataset.y]

valid_dataset_X = [x.flatten() for x in valid_dataset.X]
valid_dataset_y = [x.flatten() for x in valid_dataset.y]

#train_dataset = 


sklearn_model = KernelRidge(kernel="rbf", alpha=5e-4, gamma=0.008)


#dc_model = dc.models.SklearnModel(sklearn_model)



# Fit trained model
sklearn_model.fit(train_dataset_X, train_dataset_y)


# In[9]:


a=mean_absolute_error(
    train_dataset_y,
    sklearn_model.predict(train_dataset_X))

b=mean_absolute_error(
    valid_dataset_y,
    sklearn_model.predict(valid_dataset_X))


c=mean_absolute_error(
    test_dataset_y,
    sklearn_model.predict(test_dataset_X))

print('Normalised values (I do not know how to unnormalise this data)')

print(f'train error sklearn {a}')
print(f'valid error sklearn {b}')
print(f'test error sklearn {c}')


# ## I cannot get kernel ridge regression to work with Coulomb matrix

# In[10]:


from sklearn.kernel_ridge import KernelRidge

tasks, datasets, transformers = dc.molnet.load_qm7(
    featurizer=dc.feat.CoulombMatrix(max_atoms=23), 
    splitter='stratified', 
    move_mean=False)

train_dataset, valid_dataset, test_dataset = datasets

train_dataset, valid_dataset, test_dataset = datasets

train_dataset, valid_dataset, test_dataset = datasets

train_dataset_X = [x.flatten() for x in train_dataset.X]
train_dataset_y = [x.flatten() for x in train_dataset.y]

test_dataset_X = [x.flatten() for x in test_dataset.X]
test_dataset_y = [x.flatten() for x in test_dataset.y]

valid_dataset_X = [x.flatten() for x in valid_dataset.X]
valid_dataset_y = [x.flatten() for x in valid_dataset.y]

#dataset = dc.data.DiskDataset.from_numpy(X)
train_dataset = dc.data.datasets.DiskDataset.from_numpy(
    X=train_dataset_X, 
    y=train_dataset_y)

train_dataset = dc.data.datasets.DiskDataset.from_numpy(
    X=valid_dataset_X, 
    y=valid_dataset_y)

train_dataset = dc.data.datasets.DiskDataset.from_numpy(
    X=test_dataset_X, 
    y=test_dataset_y)

def model_builder(model_dir):
  sklearn_model = KernelRidge(kernel="rbf", alpha=5e-4, gamma=0.008)
  return dc.models.SklearnModel(sklearn_model, model_dir)

dc_model = dc.models.SklearnModel(sklearn_model)

model = dc.models.SingletaskToMultitask(
    tasks, 
    model_builder)

# Fit trained model
model.fit(train_dataset)


# In[11]:


model


# In[12]:


# little function to calc metrics on this data
out=get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers)
# makes a nice dataframe
pd_out = pd.DataFrame([out], columns=['tr_mse', 'tr_r2', 'tr_mae', 'tr_rmse',
                                        'val_mse', 'val_r2', 'val_mae', 'val_rmse',
                                        'te_mse', 'te_r2', 'te_mae', 'te_rmse'])
print(pd_out)


# In[ ]:




