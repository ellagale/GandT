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
import string
def build_char_dict():
    cti = {}
    for i, x in enumerate(string.printable):
        cti[x] = i

    cti['<unk>'] = max(cti.values()) + 1
    cti['<pad>'] = max(cti.values()) + 1
    return cti
char_to_idx=build_char_dict()

print(char_to_idx)
featurizer = dc.feat.SmilesToSeq(char_to_idx=char_to_idx)
print(str(featurizer))
print(featurizer.to_seq('CCOc1ccc2nc(S(N)(=O)=O)sc2c1'))

tasks, datasets, transformers = dc.molnet.load_qm7(
    featurizer=featurizer,
    splitter='random')

# the datasets object is already split into the train, validation and test dataset
train_dataset, valid_dataset, test_dataset = datasets
## N.B. Some molecules may not featurize and you'll get a warning this is OK

model = dc.models.Smiles2Vec(
    char_to_idx=char_to_idx,
    n_tasks=len(test_dataset.tasks), # size of y, we have one output task here: finding toxicity
    mode="regression",
    embedding_dim=len(char_to_idx)
    )

model.fit(test_dataset, nb_epoch=2)

model.predict(dataset=test_dataset,
              transformers=transformers)[:5]