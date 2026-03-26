# Included Files

# src

The actual G&T package code

## helper_functions.py

Contains the functions used by G&T

# Persistence_Notebooks

Jupyter notebooks that try out the persistence features and demonstrate the code

# Create_PHF_Scripts

A set of scripts to make persistent homology features

## Making persistent homology features

Python scripts to make some standard DeepChem datasets are included. You can read the scripts and include your own files. Note that the directories that data is read from and saved to may need to be changed for your system.

The code works by first making two csv files (x_data and y_data) of the PHF features which are saved out (and can be used in that form) before combining these into a hdf5 file.

### Preparation for making your own files

1. Make a .csv file for every column in the `database' if it is a csv file.

2. Alter the code so that the headings on the created hdf5 file is correct.

### Making the dataset

Run the files in the correct environment

# Create_Chemical_Scripts

Includes scripts to make rdkit chemical features which can be combined with PHF features.

# PCA_Chemical_Features

A series of jupyter notebooks to do PCA of chemical (rdkit) features to enable the user to pick subsets of features. Used to select features for the PDBBind experiments.

# Testing PHF datasets

Once you have made the datasets, you will want to test them and look at the data.

# Experiments

Contains jupyter notebooks to do the DeepChem experiments in Experiment 1 in the paper.

# molnet

Contains code copied from DeepChem to load and test the DeepChem datasets.