import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw
import tensorflow as tf
import os
import sys
import rdkit
import h5py

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri
import rdkit.Chem
import rdkit.Chem.AllChem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import mpl_toolkits.mplot3d
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from collections import Counter

print("TensorFlow version: " + tf.__version__)

# topology stuff
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude
from sklearn.pipeline import make_union, Pipeline

# fixc this at some point
sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

import projection
from projection.molecule import Molecule
from projection.pdbmolecule import PDBMolecule
from projection.mol2molecule import Mol2Molecule

import helper_functions as h
#from projection.face import Face

# $UN THIS
save_dir=r'F:\Nextcloud\science\Datasets\converted_pdbbind\v2015'
data_dir=r'F:\Nextcloud\science\Datasets\pdbbind\v2015'
results_dir=r"F:\Nextcloud\science\results\topology_and_graphs\PDBBind"
test_file='1a1c_pocket.pdb'
test_file_ligand='1a1c_ligand.mol2'
test_pdb_code='1a1c'
out_file_name='PDBBind_refined_topological_features.hdf5'
make_dataset=False # whether to recalc the dataset

name_file_name="INDEX_core_name.2013",
data_file_name="INDEX_core_data.2013",
cluster_file_name = "INDEX_core_cluster.2013"

df_index_core, df_data_core, df_cluster_core = h.read_in_PDBBind_data(
    data_dir,
    name_file_name="INDEX_core_name.2013",
    data_file_name="INDEX_core_data.2013",
    cluster_file_name = "INDEX_core_cluster.2013")

#RUN THIS
name_file_name="INDEX_refined_name.2013",
data_file_name="INDEX_refined_data.2013",
cluster_file_name = "INDEX_refined_cluster.2013"

df_index_refined, df_data_refined, df_cluster_refined = h.read_in_PDBBind_data(
    data_dir,
    name_file_name="INDEX_refined_name.2015",
    data_file_name="INDEX_refined_data.2015",
    cluster_file_name = "")

PDB_List=df_index_core['PDB_code']
Num_of_proteins = len(PDB_List)
feature_name_list = ['P_pers_S_1', 'P_pers_S_2', 'P_pers_S_3',
                    'P_no_p_1', 'P_no_p_2', 'P_no_p_3',
                    'P_bottle_1', 'P_bottle_2', 'P_bottle_3',
                    'P_wasser_1', 'P_wasser_2', 'P_wasser_3',
                    'P_landsc_1', 'P_landsc_2', 'P_landsc_3',
                    'P_pers_img_1', 'P_pers_img_2', 'P_pers_img_3',
                    'L_pers_S_1', 'L_pers_S_2', 'L_pers_S_3',
                    'L_no_p_1', 'L_no_p_2', 'L_no_p_3',
                    'L_bottle_1', 'L_bottle_2', 'L_bottle_3',
                    'L_wasser_1', 'L_wasser_2', 'L_wasser_3',
                    'L_landsc_1', 'L_landsc_2', 'L_landsc_3',
                    'L_pers_img_1', 'L_pers_img_2', 'L_pers_img_3'] ## finish this!

dataset = np.zeros((Num_of_proteins, 36))
i=0
for feature_name in feature_name_list:
    dataset[:,i] = fh[feature_name][:]
    i=i+1

topl_PDB_all_core_mat_large=dataset

topl_PDB_all_core_large=dataset.tolist()

topl_PDB_all_core_small = [x[0:3] for x in topl_PDB_all_core_large]
#topl_PDB_all_core_small
topl_PDB_all_core_mat_small = np.array(topl_PDB_all_core_small)
#topl_PDB_all_core_mat_small

X_data_large = topl_PDB_all_core_mat_large
X_data_small = topl_PDB_all_core_mat_small
Num_of_proteins = len(X_data)

y_dataset=np.zeros((Num_of_proteins, 2))
i=0
for feature_name in ["-logKd_over_Ki","-logKd_over_Ki"]:
    y_dataset[:,i] = fh[feature_name][:]
    i=i+1

#y_data = fh["-logKd_over_Ki"][:].tolist()

from sklearn.decomposition import PCA
pca = PCA(n_components=36)
principalComponents_large = pca.fit_transform(topl_PDB_all_core_large)
topl_PDB_all_core_small = [x[0:3] for x in topl_PDB_all_core_large]
#topl_PDB_all_core_small
topl_PDB_all_core_mat_small = np.array(topl_PDB_all_core_small)
#topl_PDB_all_core_mat_small
X_data_large=topl_PDB_all_core_large
X_data_small=topl_PDB_all_core_small

# does repeated keras NN experiments
#y_data=df_data_refined['-logKd/Ki']
y_data=df_data_core['-logKd/Ki']
X_data = X_data_small
number_of_epochs=100
num_of_proteins = len(X_data)

def run_repeated_keras_NN_tests(
    X_data,
    y_data,
    num_of_repeats=10,
    num_of_epochs=100,
    test_set_size=int(num_of_proteins*0.1),
    validate_set_size=int(num_of_proteins*0.1)):

    num_of_proteins = len(X_data)

    train_scores_r2=[]
    test_scores_r2=[]
    train_scores_rmse=[]
    test_scores_rmse=[]

    for trial in range(num_of_repeats):
        (train_X_data,
         train_y_data,
         test_X_data,
         test_y_data,
         validate_X_data,
         validate_y_data) = h.set_up_train_test_validate(
            X_data,
            y_data,
            test_set_size=test_set_size,
            validate_set_size=validate_set_size,
            verbose=True)
        # put data into a dataset for input to keras
        train_dataset=dc.data.NumpyDataset(train_X_data, train_y_data)
        test_dataset=dc.data.NumpyDataset(test_X_data, test_y_data)
        # choose metric(s)
        metric1 = dc.metrics.Metric(dc.metrics.pearson_r2_score)
        metric2 = dc.metrics.Metric(dc.metrics.rms_score)
        # make deepchem model from keras model
        model = dc.models.KerasModel(keras_model, dc.models.losses.L2Loss())
        # fit it
        model.fit(train_dataset, nb_epoch=number_of_epochs)
        train_score = model.evaluate(train_dataset, [metric1, metric2])
        test_score = model.evaluate(test_dataset, [metric1, metric2])
        train_score_r2 = train_score['pearson_r2_score']
        test_score_r2 = test_score['pearson_r2_score']
        train_score_rmse = train_score['rms_score']
        test_score_rmse = test_score['rms_score']
        print('R2 error\ttrain {:.3}, \t test: {:.3}'.format(train_score_r2, test_score_r2))
        print('RMSE error\ttrain {:.3}, \t test: {:.3}'.format(train_score_rmse, test_score_rmse))
        train_scores_r2.append(train_score_r2)
        test_scores_r2.append(test_score_r2)
        train_scores_rmse.append(train_score_rmse)
        test_scores_rmse.append(test_score_rmse)

    print(train_scores_r2)
    print(test_scores_r2)
    out=(train_scores_r2, test_scores_r2, train_scores_rmse, test_scores_rmse)
    return out

# make a model for use
keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(500, activation='relu'),
        #tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(200, activation='relu'),
        #tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(1)
    ])

out=run_repeated_keras_NN_tests(
    X_data,
    y_data,
    num_of_repeats=5,
    num_of_epochs=100,
    test_set_size=int(num_of_proteins*0.1),
    validate_set_size=int(num_of_proteins*0.1))

(train_scores_r2, test_scores_r2, train_scores_rmse, test_scores_rmse) = out