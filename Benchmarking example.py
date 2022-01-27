import deepchem
import run_benchmark
import tensorflow as tf
print(f'deepchem version:\t{deepchem.__version__}')

print(f'tensorflow version:\t{tf.__version__}')

######################### example code to try out ####################################

model_list_classification=[
      'rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv', 'dag', 'xgb',
      'weave', 'kernelsvm', 'textcnn', 'mpnn'
  ]

model_list_regression=[
      'tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg',
      'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression',
      'textcnn_regression', 'krr', 'ani', 'krr_ft', 'mpnn'
  ]

run_benchmark.run_benchmark(datasets=['qm7','qm8'], model='rf_regression', test=True)
run_benchmark.run_benchmark(datasets=['bbbp'], model='kernelsvm', test=True)

