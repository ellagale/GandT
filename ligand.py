#!/usr/bin/env python
# coding: utf-8

# # Sofia's protein stuff
#
# ### Background
#
# Nicotinic receptors, transmembrane proteisn
#
# ### The experiment
#
# Receptor minimised with ligand in place, ligand deleted (!), record how the receptor reacts. MD.
#
# ### The data
#
# Average RMSD displacements in x y z, plus sd, error. Target is efficacy of ligand (and not currently given).
#
#
#
# ### The task
#
# Does these relaxations relate to the efficacy of the ligands in the receptor in any way?
#
# To-do:
#
# set up datafames
#
# zero - start point, thesea re the PHF features:
# [[-1. -1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

# In[1]:


# lets load our libraries
# !conda install -y -c conda-forge numpy=1.19.5
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
from numpy import savetxt

print("TensorFlow version: " + tf.__version__)

# topology stuff
from gtda.mapper import plot_interactive_mapper_graph
from gtda.plotting import plot_point_cloud, plot_betti_curves
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude
from sklearn.pipeline import make_union, Pipeline

# fixc this at some point
sys.path.append(r"C:\Users\eg16993\OneDrive - University of Bristol\Documents\GitHub\GandT")
# sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

results_dir = r"C:\Users\eg16993\OneDrive - University of Bristol\Documents\Research Results\Sofia_protein"
data_dir = r"C:\Users\eg16993\OneDrive - University of Bristol\Documents\Datasets\Sofia_protein_data"

# import projection
# from projection.molecule import Molecule
# from projection.pdbmolecule import PDBMolecule
# from projection.mol2molecule import Mol2Molecule

from src import helper_functions as h


# from projection.face import Face


# In[2]:


def coords_to_persistence_diagrams(coords):
    # makes a point cloud version of the structure
    # there are no atom types

    # Track connected components, loops, and voids
    homology_dimensions = [0, 1, 2]

    # Collapse edges to speed up H2 persistence calculation!
    persistence = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=homology_dimensions,
        n_jobs=6,
        collapse_edges=True,
    )
    reshaped_coords = coords[None, :, :]
    diagrams_basic = persistence.fit_transform(reshaped_coords)
    return coords, diagrams_basic


def sort_input_files(current_dir):
    # sorts files in numerical order
    file_list = os.listdir(current_dir)
    file_list = [x for x in file_list if x[0] == 'a']
    file_list.sort()
    file_dict = {}
    for file in file_list:
        file_dict[int(file.split('_')[3].split('p')[0])] = file
    timestamp_list = [x for x in file_dict.keys()]
    timestamp_list.sort()
    file_list = [file_dict[timestamp] for timestamp in timestamp_list]
    file_list

    return file_list, timestamp_list


# In[3]:


################################################################################################################################################################################################
#                 Average Displacement                Sample Size                                   SE (1SD)                SE (1SD) of              SE (2SD)                SE (2SD) of       #
#              --------------------------       ---------------------      Average        --------------------------          Average        --------------------------          Average       #
#       CA     x-axis    y-axis    z-axis       x         y         z    Displacement     x-axis    y-axis    z-axis    Displacement vector  x-axis    y-axis    z-axis    Displacement vector #
################################################################################################################################################################################################


# In[3]:


ligand_list = ['lig1', 'lig2', 'lig3', 'lig4', 'lig5']
for ligand_name in ligand_list:
    current_dir = os.path.join(data_dir, ligand_name)
    file_list, timestamp_list = sort_input_files(current_dir)
    print(len(timestamp_list))

# In[ ]:


ligand_list = ['lig1', 'lig2', 'lig3', 'lig4', 'lig5']
# ligand_list = ['lig2','lig3']#,'lig3','lig4','lig5']

verbose = True
topol_feat_list = []
trajectory_diagram_list = []

# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

######## make pipeline for the topological calculations #####

pipe = h.make_pipeline(num_of_features=18)
max_value_per_lig = []

for ligand_name in ligand_list:

    current_dir = os.path.join(data_dir, ligand_name)

    headings = ['CA', 'x', 'y', 'z', 'Sx', 'Sy', 'Sz', 'AveD', 'SE_x', 'SE_y', 'SE_z', 'SE_AveD', 'SE_x2', 'SE_y2',
                'SE_z2', 'SE_AveD2']

    file_list, timestamp_list = sort_input_files(current_dir)

    trajectory = []
    ##### reads files
    for filename in file_list:
        # print(filename)
        # read file
        with open(os.path.join(current_dir, filename)) as f:
            lines = [line.rstrip('\n') for line in f]
        # find coords
        coords = []
        for i in range(len(lines)):
            line = lines[i]
            if i < 6:
                # print(line)
                continue
            else:

                egg = [x for x in line.split(' ') if not x == '']

                coords.append([float(egg[1]), float(egg[2]), float(egg[3])])

        coords = np.array(coords)
        trajectory.append(coords)

    ##### does the persistent homology
    diagrams_list = []
    ##### does zero

    # print(f'Doing snapshot {snapshot} for {ligand_name}')
    coords = trajectory[0]
    coords, diagrams_basic = coords_to_persistence_diagrams(coords)

    X_basic = pipe.fit_transform(diagrams_basic)

    topol_feat_list.append([x for x in X_basic[0]])
    if verbose:
        print(X_basic)
    ##### saves trajectory features

    # savetxt('zero_' + ligand_name + '_PHF_features.csv', topol_feat_mat, delimiter=',')

    fig = plot_diagram(diagrams_basic)

    for snapshot in range(1, 3): #len(timestamp_list)):
        print(f'Doing snapshot {snapshot} for {ligand_name}')
        coords = trajectory[snapshot]
        # coords, diagrams_basic=coords_to_persistence_diagrams(coords)

        # Collapse edges to speed up H2 persistence calculation!
        persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=homology_dimensions,
            n_jobs=6,
            collapse_edges=True,
        )
        reshaped_coords = coords[None, :, :]
        diagrams_basic = persistence.fit_transform(reshaped_coords)
        diagrams_list.append(diagrams_basic[0])
        X_basic = pipe.fit_transform(diagrams_basic)

        topol_feat_list.append([x for x in X_basic[0]])
        if verbose:
            print(X_basic)
    ##### saves trajectory features
    topol_feat_mat = np.array(topol_feat_list)
    savetxt(ligand_name + '_PHF_features.csv', topol_feat_mat, delimiter=',')

    #### need to save diagrams list somehow

    # get max value so the images are scaled nicely
    max_x = []
    max_y = []

    max_value = 0

    for i in range(len(diagrams_list)):
        current_diagram = diagrams_list[:][i]
        max_x.append(current_diagram[:, 0].max())
        max_y.append(current_diagram[:, 0].max())

    max_value = np.array(max_x).max()
    max_value_per_lig.append(max_value)
    max_value = max_value + 0.07 * max_value

    ###### save figures
    for snapshot in range(0, len(diagrams_list)):
        fig = plot_diagram(diagrams_list[snapshot])
        fig.update_xaxes(range=[-0.005, max_value])
        fig.update_yaxes(range=[-0.005, max_value])
        # fig.show()
        fig.write_image(ligand_name + '_' + str(timestamp_list[snapshot]) + 'ps.png')

    trajectory_diagram_list.append(diagrams_list)
    ##### end ligand

#### scaled images


max_value_scaled = np.array(max_value_per_lig).max()
max_value_scaled = max_value_scaled + 0.07 * max_value_scaled
for i in range(len(ligand_list)):
    ligand_name = ligand_list[i]
    diagrams_list = trajectory_diagram_list[i]

    # save figures
    for snapshot in range(1, len(diagrams_list)):
        fig = plot_diagram(diagrams_list[snapshot])
        fig.update_xaxes(range=[-0.005, max_value_scaled])
        fig.update_yaxes(range=[-0.005, max_value_scaled])
        #   fig.show()
        fig.write_image('scaled_' + ligand_name + '_' + str(timestamp_list[snapshot]) + 'ps_scaled.png')
    # does zero
    coords = trajectory[0]
    coords, diagrams_basic = coords_to_persistence_diagrams(coords)
    fig = plot_diagram(diagrams_basic)

    fig.update_xaxes(range=[-0.005, max_value_scaled])
    fig.update_yaxes(range=[-0.005, max_value_scaled])
    #   fig.show()
    fig.write_image('scaled_' + ligand_name + '_' + str(0) + 'ps_scaled.png')

# In[ ]:


# pip install -U kaleido


# In[ ]:

