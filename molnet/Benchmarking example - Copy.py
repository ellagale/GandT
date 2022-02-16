import deepchem
from deepchem.molnet import run_benchmark
import tensorflow as tf
print(f'deepchem version:\t{deepchem.__version__}')

print(f'tensorflow version:\t{tf.__version__}')

######################### example code to try out ####################################

model_list_classification=[
      'rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv', 'dag', 'xgb',
      'weave', 'kernelsvm', 'textcnn', 'mpnn'
  ]

classification_datasets = [
        'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'pcba_146',
        'pcba_2475', 'sider', 'tox21', 'toxcast', 'bace_c'
    ]

model_list_regression=[
      'tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg',
      'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression',
      'textcnn_regression', 'krr', 'ani', 'krr_ft', 'mpnn'
  ]

regression_datasets= [
        'bace_r', 'chembl', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo',
        'nci', 'pdbbind', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl',
        'thermosol'
    ]

classification_datasets = [
        'bbbp', 'tox21', 'clintox', 'hiv', 'sider',  'toxcast', 'bace_c'
    ]

regression_datasets= [
         'qm7','qm8','delaney', 'lipo', 'pdbbind', 'qm7b', 'qm9'
    ]

metric_dict={'qm7': deepchem.metrics.mae_score,
             'qm7b': deepchem.metrics.mae_score,
             'qm8': deepchem.metrics.mae_score,
             'qm9': deepchem.metrics.mae_score,
             'delaney': deepchem.metrics.rms_score,
             'lipo': deepchem.metrics.rms_score,
             'pdbbind': deepchem.metrics.rms_score}

run_regression=False
if run_regression==True:
    for dataset in regression_datasets:
        for model in model_list_regression:
            metric=metric_dict[dataset]
            try:
                run_benchmark.run_benchmark([dataset],
                      model,
                      split=None,
                      metric=metric, # use errors not r2
                      direction=False, # want to minimise the error
                      featurizer=None, # use default
                      n_features=0,
                      out_path='.',
                      hyper_parameters=None, # use defaults
                      hyper_param_search=False, # use defaults
                      max_iter=20,
                      search_range=2,
                      test=True,
                      reload=True,
                      seed=123)
            except:
                continue

for dataset in classification_datasets:
    for model in model_list_classification:
        #metric=metric_dict[dataset]
        print(f"dataset:\t{dataset}, model\t{model}")
        try:
            run_benchmark.run_benchmark([dataset],
                  model,
                  split=None,
                  metric=None, # will AUC-ROC
                  direction=True, # want to maximise the AUC
                  featurizer=None, # use default
                  n_features=0,
                  out_path='.',
                  hyper_parameters=None, # use defaults
                  hyper_param_search=False, # use defaults
                  max_iter=20,
                  search_range=2,
                  test=True,
                  reload=True,
                  seed=123)
        except:
            continue



