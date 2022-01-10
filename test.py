import sys

sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

import deepchem as dc

import tensorflow as tf
import sys

print("TensorFlow version: " + tf.__version__)

# topology stuff

# fixc this at some point
sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

# from projection.face import Face

# $UN THIS
data_dir = r'F:\Nextcloud\science\Datasets\topol_datasets'
results_dir = r"F:\Nextcloud\science\results\topology_and_graphs\QM7"
test_file = 'qm7.csv'
data_file_name = 'qm7_topological_features.hdf5'
make_dataset = False  # whether to recalc the dataset

print(f"DeepChem version: {dc.__version__}")

############################### settings for all experiments #################

num_repeats = 3
num_epochs = 3

metric_labels = ['mean_squared_error', 'pearson_r2_score',
                 'mae_score', 'rmse']

metric1 = dc.metrics.Metric(dc.metrics.mean_squared_error)
metric2 = dc.metrics.Metric(dc.metrics.pearson_r2_score)
metric3 = dc.metrics.Metric(dc.metrics.mae_score)
metrics = [metric1, metric2, metric3]
selected_metric = 2  # which metric to use for callback
featurizer = dc.feat.ConvMolFeaturizer()
import pandas
print(pandas.__version__)
import numexpr
print(numexpr.__version__)
tasks, datasets, transformers_gc = dc.molnet.load_qm7(
    shard_size=2000, featurizer=featurizer, splitter="stratified")
train_dataset, valid_dataset, test_dataset = datasets
n_tasks = len(tasks)
model = dc.models.GraphConvModel(n_tasks, mode='regression')
model.fit(train_dataset, nb_epoch=50)
