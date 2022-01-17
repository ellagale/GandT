# ######################################################################################
#           Helper functions for graphs and topology in deepchem                       #
########################################################################################
import os
from collections import Counter

import deepchem as dc
import h5py
import numpy as np
import pandas as pd
import rdkit
import tensorflow as tf
from deepchem.feat.base_classes import UserDefinedFeaturizer
from gtda.diagrams import Amplitude
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import PersistenceEntropy
# topology stuff
from gtda.homology import VietorisRipsPersistence
from rdkit.Chem import AllChem
from rdkit import Chem
from scipy.spatial.transform import Rotation as R
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_union, Pipeline

import matplotlib as plt

num_of_molecules = 100

results_dir=os.getcwd()


def coord_getter(data_dir, test_file, test_pdb_code, setting='pdb'):
    """New quick function to grab the coordinates from the files
    Doesn't do all the conformer and rotation stuff that the functions in
    projection does"""
    test_file_location = os.path.join(data_dir, test_pdb_code, test_file)
    if setting == 'pdb':
        mol_orig = rdkit.Chem.rdmolfiles.MolFromPDBFile(test_file_location)
    elif setting == 'mol2':
        mol_orig = rdkit.Chem.rdmolfiles.MolFromMol2File(
            test_file_location,
            cleanupSubstructures=False,
            sanitize=False)
    egg = mol_orig.GetConformer()
    rdkit.Chem.rdMolTransforms.CanonicalizeConformer(egg)
    coords = egg.GetPositions()
    return mol_orig, coords

def coord_creator(smiles):
    """Calculates coords form smiles strings"""
    #test_file_location = os.path.join(data_dir, test_pdb_code, test_file)
    mol_orig = Chem.MolFromSmiles(smiles)
    mol_orig = Chem.AddHs(mol_orig)

    status = AllChem.EmbedMolecule(mol_orig)
    status = AllChem.UFFOptimizeMolecule(mol_orig)
    egg = mol_orig.GetConformer()
    rdkit.Chem.rdMolTransforms.CanonicalizeConformer(egg)
    coords = egg.GetPositions()
    return mol_orig, coords


def coord_getter_2(file_location, setting='pdb'):
    print(file_location)
    if setting == 'pdb':
        mol_orig = rdkit.Chem.rdmolfiles.MolFromPDBFile(file_location, sanitize=False)
        # rdkit.Chem.rdMolTransforms.CanonicalizeConformer(self.conformer)
    elif setting == 'mol2':
        mol_orig = rdkit.Chem.rdmolfiles.MolFromMol2File(
            file_location,
            cleanupSubstructures=False,
            sanitize=False)
    egg = mol_orig.GetConformer()
    rdkit.Chem.rdMolTransforms.CanonicalizeConformer(egg)
    coords = egg.GetPositions()
    return mol_orig, coords


def read_in_PDBBind_data(
        data_dir,
        name_file_name="INDEX_core_name.2013",
        data_file_name="INDEX_core_data.2013",
        cluster_file_name="INDEX_core_cluster.2013"):
    # if True:

    """script to read into the data for PDBBind"""

    # reads in the name data
    if not name_file_name == '':
        ## This reads in the pdb codes for each protein in the core dataset
        fh = open(os.path.join(data_dir, name_file_name), 'r')
        c = 0
        column_list_name = ["PDB_code", "release_year", "EC_number", "protein_name"]
        lines = []
        for line in fh.readlines():
            if not line.startswith('#'):
                if c == 0:
                    words = [x for x in line.split(' ') if not x == '']
                    last_word = words[3:]
                    last_word[-1] = last_word[-1].strip()
                    last_word = '_'.join(last_word)
                    new_line = [words[0], words[1], words[2], last_word]
                    # print(new_line)
                    lines.append(new_line)
                    # df_line = pd.DataFrame([new_line],columns=column_list)
                    # print(df_line)
                    # df_index_core.append(df_line)
                    # labels = line

        fh.close()
        df_index_core = pd.DataFrame(lines, columns=column_list_name)
        print('Sample of name file:')
        print(df_index_core.head())
    else:
        df_index_core = ''
    # reads in the data, including the y values
    if not data_file_name == '':
        ## This reads in the data for each protein core dataset
        ## the y values is the -logKd/Ki
        fh = open(os.path.join(data_dir, data_file_name), 'r')
        c = 0
        column_list_data = ["PDB_code",
                            "resolution",
                            "release_year",
                            "-logKd/Ki",
                            "Kd/Ki",
                            "reference",
                            "ligand name"]
        lines = []
        for line in fh.readlines():
            if not line.startswith('#'):
                if c == 0:
                    words = [x for x in line.split(' ') if not x == '']
                    # Not actually the last word!
                    last_word = words[5:-1]
                    last_word[-1] = last_word[-1].strip()
                    last_word = '_'.join(last_word)

                    new_line = [words[0], words[1], words[2], words[3], words[4], last_word, words[-1][1:-2]]
                    # print(new_line)
                    lines.append(new_line)
                    # df_line = pd.DataFrame([new_line],columns=column_list)
                    # print(df_line)
                    # df_index_core.append(df_line)
                    # labels = line

        fh.close()
        df_data_core = pd.DataFrame(lines, columns=column_list_data)
        df_data_core.head()
        print('Sample of data file:')
        print(df_data_core.head())
    else:
        df_data_core = ''
    # reads int hte cluster data, this is only available for the core dataset
    if not cluster_file_name == '':
        ## This reads in the cluster data for each protein in the core dataset
        ## the y values is the -logKd/Ki
        fh = open(os.path.join(data_dir, cluster_file_name), 'r')
        c = 0
        column_list_cluster = ['PDB_code',
                               'resolution',
                               'release_year',
                               '-logKd/Ki',
                               'original_Kd/Ki',
                               'cluster ID']
        lines = []
        for line in fh.readlines():
            if not line.startswith('#'):
                if c == 0:
                    words = [x for x in line.split(' ') if not x == '']
                    last_word = words[5:]
                    last_word[-1] = last_word[-1].strip()
                    last_word = '_'.join(last_word)
                    new_line = [words[0], words[1], words[2], words[3], words[4], last_word]
                    # print(new_line)
                    lines.append(new_line)
                    # df_line = pd.DataFrame([new_line],columns=column_list)
                    # print(df_line)
                    # df_index_core.append(df_line)
                    # labels = line

        fh.close()
        df_cluster_core = pd.DataFrame(lines, columns=column_list_cluster)
        print('Sample of cluster file:')
        print(df_cluster_core.head())
    else:
        df_cluster_core = ''
    return df_index_core, df_data_core, df_cluster_core


def make_topological_features_for_PDBBind(df_cluster_core,
                                          structure_file_format='mol2',
                                          verbose=False,
                                          Num_of_proteins=0,
                                          Num_of_features=3,
                                          data_dir='',
                                          do_specified_range=False,
                                          selected_range=[]):
    """makes topological features from pdb and mol2 files in PDBBind
    df_cluster_core = dataframe of the cluster dataset
    structure_file_format='mol2': pdb fro proteins, mol2 for ligands
    Num_of_proteins=0 nume of proteins to do, 0 is all
    Num_of_features: 3 for persistence entropy only, 9 for everything
    do_specified_range: whether you will give the actual range
    range: the range as a vector of indices
    """
    if True:
        if not do_specified_range:
            # range is false
            if Num_of_proteins == 0:
                # does all proteins at once
                Num_of_proteins = len(df_cluster_core)
            selected_range = [x for x in range(Num_of_proteins)]

        if do_specified_range:
            Num_of_proteins = len(selected_range)

    point_ptr = -1
    # structure_file_format='pdb'
    pdb_list = df_cluster_core['PDB_code']
    if structure_file_format == 'mol2':
        input_file_end_name = 'ligand'
    elif structure_file_format == 'pdb':
        input_file_end_name = 'pocket'

    topol_feat_list = []
    if Num_of_features == 18:
        # makes 18 topological features
        # To-DO make it possible to select a subset if you are bothered
        # Select a variety of metrics to calculate amplitudes
        # makes a pipeline
        metrics = [
            {"metric": metric}
            for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
        ]

        # Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
        feature_union = make_union(
            PersistenceEntropy(normalize=True),
            NumberOfPoints(n_jobs=-1),
            *[Amplitude(**metric, n_jobs=-1) for metric in metrics]
        )

        ## then we use a pipeline to transform, the data and spit i out
        # mwah hahahahaha
        pipe = Pipeline(
            [
                ("features", feature_union)
            ]
        )
        print('Doing 18 (3 x 6) topology features:')
        print('Persistence entropy\nnumber of points')
        for m in metrics:
            print(m)
    else:
        print('Doing 3 features, persistence entropy only')

    for mol_idx in selected_range:
        print('Got to Molecule no. ', mol_idx)
        ### load da data
        file_location = os.path.join(data_dir,
                                     pdb_list[mol_idx],
                                     pdb_list[mol_idx] + '_' + input_file_end_name + '.' + structure_file_format)
        if structure_file_format == 'mol2':
            mol, coords = coord_getter_2(file_location, setting='mol2')
        elif structure_file_format == 'pdb':
            mol, coords = coord_getter_2(file_location, setting='pdb')
        elif structure_file_format == 'smiles': # no file to read!
            mol, coords = coord_creator(smiles)

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
        persistence_entropy = PersistenceEntropy()
        if Num_of_features == 3:
            X_basic = persistence_entropy.fit_transform(diagrams_basic)
        elif Num_of_features == 18:
            X_basic = pipe.fit_transform(diagrams_basic)
        topol_feat_list.append([x for x in X_basic[0]])
        if verbose:
            print(X_basic)

    topol_feat_mat = np.array(topol_feat_list)

    return topol_feat_list, topol_feat_mat


def create_and_merge_PDBBind_topol_features(df_cluster_core,
                                            Num_of_proteins=5,
                                            Num_of_features=18,
                                            data_dir='',
                                            do_specified_range=False,
                                            selected_range=[],
                                            verbose=False):
    """merges topological features for ligand nad protein into the same dataset
    i.e. each row has x protein features and y ligand features"""
    # grab ligands
    print('Doing the proteins...')
    topol_feat_list_protein, topol_feat_mat_protein = make_topological_features_for_PDBBind(df_cluster_core,
                                                                                            structure_file_format='pdb',
                                                                                            verbose=verbose,
                                                                                            Num_of_proteins=Num_of_proteins,
                                                                                            Num_of_features=Num_of_features,
                                                                                            data_dir=data_dir,
                                                                                            do_specified_range=do_specified_range,
                                                                                            selected_range=selected_range)

    # do proteins
    print('Doing the ligands')
    topol_feat_list_ligand, topol_feat_mat_ligand = make_topological_features_for_PDBBind(df_cluster_core,
                                                                                          structure_file_format='mol2',
                                                                                          verbose=verbose,
                                                                                          Num_of_proteins=Num_of_proteins,
                                                                                          Num_of_features=Num_of_features,
                                                                                          data_dir=data_dir,
                                                                                          do_specified_range=do_specified_range,
                                                                                          selected_range=selected_range)
    # list version of data
    topl_PDB_all_core = [topol_feat_list_protein[i] + topol_feat_list_ligand[i] for i in
                         range(len(topol_feat_list_ligand))]

    # numpy version of data
    topl_PDB_all_core_mat = np.array(topl_PDB_all_core)
    np.savetxt("test.txt", topl_PDB_all_core_mat)
    if verbose:
        print(topl_PDB_all_core)
        print(topl_PDB_all_core_mat)
    return (topl_PDB_all_core, topl_PDB_all_core_mat)


def nice_stats_outputter(train_scores, test_scores, validate_scores=''):
    """does mean min max and standard error"""

    min_train = np.min(train_scores)
    mean_train = np.mean(train_scores)
    std_error_train = np.std(train_scores) / np.sqrt(len(train_scores))
    max_train = np.max(train_scores)

    print('training: \t {:.3} <\t {:.3}. +/- {:.3} \t< {:.3} '.format(
        min_train,
        mean_train,
        std_error_train,
        max_train))
    if not validate_scores == '':
        min_validate = np.min(validate_scores)
        mean_validate = np.mean(validate_scores)
        std_error_validate = np.std(validate_scores) / np.sqrt(len(validate_scores))
        max_validate = np.max(validate_scores)
        print('validation: \t {:.3} <\t {:.3}. +/- {:.3} \t< {:.3} '.format(
            min_validate,
            mean_validate,
            std_error_validate,
            max_validate))
    min_test = np.min(test_scores)
    mean_test = np.mean(test_scores)
    std_error_test = np.std(test_scores) / np.sqrt(len(test_scores))
    max_test = np.max(test_scores)
    print('testing: \t {:.3} <\t {:.3}. +/- {:.3} \t< {:.3} '.format(
        min_test,
        mean_test,
        std_error_test,
        max_test))
    if not validate_scores == '':
        out = (min_train, mean_train, std_error_train, max_train,
               min_validate, mean_validate, std_error_validate, max_validate,
               min_test, mean_test, std_error_test, max_test)
    else:
        out = (min_train, mean_train, std_error_train, max_train,
               min_test, mean_test, std_error_test, max_test)
    return out


def set_up_train_test_validate(
        X_data,
        y_data,
        test_set_size=None,
        validate_set_size=None,
        verbose=False):
    """Calcs the indices for train/test/validate split
    X_data=X data as np array
    y_data=y dataframe
    test_set_size=int(num_of_proteins*0.1),
    validate_set_size=int(num_of_proteins*0.1),
    verbose=False"""

    num_of_proteins = len(X_data)
    if test_set_size == None:
        if verbose:
            print('Assuming a 10% test set')
            test_set_size = int(0.1 * num_of_proteins)
            print('test set has {} items'.format(test_set_size))
    if validate_set_size == None:
        if verbose:
            print('Assuming a 10% validation set')
            validate_set_size = int(0.1 * num_of_proteins)
            print('test set has {} items'.format(validate_set_size))

    test_data_indices = np.random.choice(num_of_proteins, test_set_size, replace=False)
    validate_data_indices = np.random.choice(num_of_proteins, validate_set_size, replace=False)
    not_train_indices = np.concatenate((test_data_indices, validate_data_indices))
    # print(test_data_indices)
    test_X_data = [X_data[i] for i in test_data_indices]
    validate_X_data = [X_data[i] for i in validate_data_indices]
    train_X_data = [X_data[i] for i in range(num_of_proteins) if i not in not_train_indices]
    y = y_data[:num_of_proteins]

    def get_row_either(data, row):
        """
        mini helper function to handle the fact you can't always do [] dereferencing on dataframes.
        """
        try:
            return data[row]
        except KeyError:
            return data.iloc[row]

    if len(y.shape) > 1:
        test_y_data = [np.array(get_row_either(y, i), dtype='f4') for i in test_data_indices]
        validate_y_data = [np.array(get_row_either(y, i), dtype='f4') for i in validate_data_indices]
        train_y_data = [np.array(get_row_either(y, i), dtype='f4') for i in range(num_of_proteins) if
                        i not in not_train_indices]
    else:
        # 1D Y Data. Ez
        test_y_data = [float(y[i]) for i in test_data_indices]
        validate_y_data = [float(y[i]) for i in validate_data_indices]
        train_y_data = [float(y[i]) for i in range(num_of_proteins) if i not in not_train_indices]
    if verbose:
        print('Test set indices:')
        print(test_data_indices)
        print('Validation set indices:')
        print(validate_data_indices)
    return (train_X_data, train_y_data, test_X_data, test_y_data, validate_X_data, validate_y_data)


def create_or_recreate_dataset(fh, datasetname, shape, dtype):
    """
    Create a dataset with the given parameters in the file pointed to by fh
    if the dataset already exists then it will be deleted first.
    """
    try:
        del fh[datasetname]
    except KeyError:
        pass
    return fh.create_dataset(datasetname, shape, dtype)


def Open_Train_File_Create_Datasets(data_dir,
                                    outfile,
                                    field,
                                    norm_L2_field,
                                    norm_mean_field,
                                    norm_std_field,
                                    label='molID'):
    """Function to open the dataset, make new datasets to populate later,
    grab the data to be normalised and return the file handle
    N.B. does not shut the file"""
    # Open, calc basic details
    print(outfile)
    fh = h5py.File(os.path.join(data_dir, outfile), 'r+')
    num_of_rows, num_of_molecules, = basic_info_hdf5_dataset(fh, label=label)

    # grabs the original data
    data = fh[field]
    row_shape = data.shape[1:]
    # makes a new dataset of the same size
    data_L2_out = create_or_recreate_dataset(fh, norm_L2_field, data.shape, data.dtype)
    data_mean_out = create_or_recreate_dataset(fh, norm_mean_field, data.shape, data.dtype)
    data_std_out = create_or_recreate_dataset(fh, norm_std_field, data.shape, data.dtype)

    row_count = data.shape[0]
    if not num_of_rows == row_count:
        print('Error: num_of_rows from molID is not the same as the number found for {field}')
    print(f'row_shape is {row_shape}')
    return fh, data, data_L2_out, data_mean_out, data_std_out


def basic_info_hdf5_dataset(hf, label='molID'):
    """Calcs some basic data
    hf is the file handle to hdf5 file
    label is the unique ID label per class/molecule in hte dataset"""
    molID_List_orig = hf[label]
    num_of_rows = len(molID_List_orig)
    print(f'num_of_ rows is:\t{num_of_rows}')
    counted_molID_List = Counter(molID_List_orig)
    molID_List = [x for x in counted_molID_List.keys()]
    # print(molID_List)
    num_of_molecules = len(molID_List)
    print(f'num_of_molecules is:\t {num_of_molecules}')
    egg = Counter(Counter(molID_List_orig).values())
    if not len(egg.keys()) == 1:
        print('Warning: Unbalanced dataset\nMolID: count')
        print(counted_molID_List)
    else:
        num_of_augments = [x for x in egg.keys()][0]
        print(f'num_of_augments is:\t{num_of_augments}')
    return num_of_rows, num_of_molecules


########################## deep chem data grabber helper functions #######################

class My_Dummy_Featurizer(UserDefinedFeaturizer):
    """Currently the dummy featurizer in DeepChem is broken
    This does the same thing and will load the information from the
    MoleculeNet datasets into rdkit molecules! Yay!"""

    def _featurize(self, datapoint):
        return datapoint


def coord_getter_from_deepchem(mol):
    coords = mol.GetConformer().GetPositions()
    return mol, coords


def make_pipeline(num_of_features):
    """Sets up a pipeline to create the features
    currently you have no choice over metrics"""
    if num_of_features == 18:
        # makes pipeline if you're doing 18 topol features
        # just returns persistence entropy if you're only doing 3
        # makes 18 topological features
        metrics = [
            {"metric": metric}
            for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
        ]

        # Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
        feature_union = make_union(
            PersistenceEntropy(normalize=True),
            NumberOfPoints(n_jobs=-1),
            *[Amplitude(**metric, n_jobs=-1) for metric in metrics]
        )

        ## then we use a pipeline to transform, the data and spit i out
        # mwah hahahahaha
        pipe = Pipeline(
            [
                ("features", feature_union)
            ]
        )
        print('Doing 18 (3 x 6) topology features:')
        print('Persistence entropy\nnumber of points')
        for m in metrics:
            print(m)
    else:
        print('Doing 3 features, persistence entropy only')
        pipe = PersistenceEntropy()
    return pipe


def make_topological_features_from_deepchem(my_dataset,
                                            file_type='dc',
                                            verbose=False,
                                            num_of_molecules=0,
                                            num_of_features=3,
                                            data_dir='',
                                            do_specified_range=False,
                                            selected_range=[],
                                            pdb_list=[],
                                            input_file_end_name='',
                                            skip_molecules=[]):
    """makes topological features from pdb and mol2 files in PDBBind
    df_cluster_core = dataframe of the cluster dataset
    structure_file_format='mol2': pdb fro proteins, mol2 for ligands
    Num_of_proteins=0 nume of proteins to do, 0 is all
    Num_of_features: 3 for persistence entropy only, 9 for everything
    do_specified_range: whether you will give the actual range
    range: the range as a vector of indices
    N.B. pdb and mol2 file type not tested
    """
    ########### set up what you're going to run ################
    if not do_specified_range:
        # range is false
        if num_of_molecules == 0:
            # does all proteins at once
            num_of_molecules = len(my_dataset)
        selected_range = [x for x in range(num_of_molecules)]

    if do_specified_range:
        num_of_molecules = len(selected_range)

    point_ptr = -1
    # structure_file_format='pdb'

    topol_feat_list = []

    ######## make pipeline for the topological calculations #####

    pipe = make_pipeline(num_of_features)

    ############# DAS LOOP! Makes topological features ##########

    for mol_idx in selected_range:
        if not mol_idx in skip_molecules:
            print('Got to Molecule no. ', mol_idx)
            ### load da data from dataset is rdkit mol object already
            mol = my_dataset.X[mol_idx]
            if file_type == 'dc':
                _, coords = coord_getter_from_deepchem(mol)
            else:
                # data is in files somewhere we make the rdkit mol object
                if not file_type == 'smiles':
                    file_location = os.path.join(data_dir,
                                             pdb_list[mol_idx],
                                             pdb_list[mol_idx] + '_' + input_file_end_name + '.' + file_type)
                if file_type == 'mol2':
                    mol, coords = coord_getter_2(file_location, setting='mol2')
                elif file_type == 'pdb':
                    mol, coords = coord_getter_2(file_location, setting='pdb')
                elif file_type == 'smiles':
                    mol, coords = coord_creator(mol)

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
            # persistence_entropy = PersistenceEntropy()
            # if num_of_features == 3:
            X_basic = pipe.fit_transform(diagrams_basic)
            # elif Num_of_features == 18:
            # X_basic = pipe.fit_transform(diagrams_basic)
            topol_feat_list.append([x for x in X_basic[0]])
            if verbose:
                print(X_basic)

        topol_feat_mat = np.array(topol_feat_list)

    return topol_feat_list, topol_feat_mat

######################### functions to do experiments ! ##########################

def run_repeated_RF_tests(
        X_data,
        y_data,
        num_of_repeats=10,
        num_of_estimators=100,
        test_set_size=int(num_of_molecules * 0.1),
        validate_set_size=int(num_of_molecules * 0.1)):
    """Runs some RF baselines on input data
    expects np.arrays &/or dataframes for X and y
    (yes I need to test it!)"""

    num_of_molecules = len(X_data)

    train_scores = []
    test_scores = []

    for trail in range(num_of_repeats):
        # this sets up the indices for train/test/validate
        (train_X_data,
         train_y_data,
         test_X_data,
         test_y_data,
         validate_X_data,
         validate_y_data) = set_up_train_test_validate(
            X_data,
            y_data,
            test_set_size=test_set_size,
            validate_set_size=validate_set_size)
        model = RandomForestRegressor(n_estimators=num_of_estimators)
        model.fit(train_X_data, train_y_data)
        train_scores.append(model.score(train_X_data, train_y_data))
        test_scores.append(model.score(test_X_data, test_y_data))

    return (train_scores, test_scores, model)

def create_keras_model(size_of_output = 1):
    """Makes a simple NN for testing"""
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
    tf.keras.layers.Dense(size_of_output)
    ])
    return keras_model

def run_repeated_keras_NN_tests(
    X_data,
    y_data,
    keras_model=None,
    num_of_repeats=10,
    num_of_epochs=100,
    test_set_size=int(num_of_molecules*0.1),
    validate_set_size=int(num_of_molecules*0.1)):

    num_of_proteins = len(X_data)
    size_of_output = len(y_data.columns)

    train_scores_r2=[]
    test_scores_r2=[]
    train_scores_rmse=[]
    test_scores_rmse=[]

    for trial in range(num_of_repeats):
        (train_X_data, ###
         train_y_data,
         test_X_data,
         test_y_data,
         validate_X_data,
         validate_y_data) = set_up_train_test_validate(
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
        if keras_model == None:
            keras_model = create_keras_model(size_of_output=size_of_output)
        model = dc.models.KerasModel(keras_model, dc.models.losses.L2Loss())
        # fit it
        model.fit(train_dataset, nb_epoch=num_of_epochs)
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
    out=(train_scores_r2, test_scores_r2, train_scores_rmse, test_scores_rmse, keras_model)
    return out

        # make a model for use

def create_and_merge_dc_topol_features(my_dataset,
                                       num_of_molecules=5,
                                       num_of_features=18,
                                       data_dir='',
                                       save_dir='',
                                       do_specified_range=False,
                                       selected_range=[],
                                       file_type='dc',
                                       verbose=False,
                                       skip_molecules=[]):
    """merges topological features for deep chem loaded datasets into the same dataset
    i.e. each row has x protein features and y ligand features"""
    # grab ligands
    print('Doing the molecules...')

    topol_feat_list, topol_feat_mat = make_topological_features_from_deepchem(my_dataset,
                                                                                file_type=file_type,
                                                                                verbose=False,
                                                                                num_of_molecules=num_of_molecules,
                                                                                num_of_features=num_of_features,
                                                                                data_dir=data_dir,
                                                                                do_specified_range=do_specified_range,
                                                                                selected_range=selected_range,
                                                                                skip_molecules=skip_molecules)

    # numpy version of data
    np.savetxt(os.path.join(save_dir,"test.txt"), topol_feat_mat)
    if verbose:
        print(topol_feat_mat)
        print(topol_feat_list)
    return (topol_feat_list, topol_feat_mat)

def temp_write_topol_data(
        f,
        remaining,
        current_ptr,
        my_dataset,
        num_of_topol_features,
        do_specified_range,
        batch_size,
        data_dir,
        file_type='dc',
        skip_molecules=[]
):
    """wrapper function to create_and_merge_dc_topol_features"""
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        this_range = list(range(current_ptr, current_ptr + this_batch))

        out_list, y_new = create_and_merge_dc_topol_features(my_dataset,
                                                             num_of_molecules=5, # overwritten by do_specified_Range
                                                             num_of_features=num_of_topol_features,
                                                             data_dir=data_dir,
                                                             do_specified_range=do_specified_range,
                                                             selected_range=this_range,
                                                             file_type=file_type,
                                                             verbose=False,
                                                             skip_molecules=skip_molecules)

        for _list in out_list:
            row = ','.join([str(i) for i in _list])
            f.write(row + '\n')
        remaining -= this_batch
        current_ptr += this_batch
    return

def copy_targets_into_csv(
        y_fh,
        my_dataset
):
    """writes targets from dataset to file handle for csv file output"""
    for target_list in my_dataset:
        if len(target_list) == 1:
            # single target
            row = str(target_list[0])
        else:  # multidimensional target
            row = ','.join([str(i) for i in target_list])
        y_fh.write(row + '\n')
    return

def load_all_hdf5(fh,
                  num_of_rows,
                  column_headers):
    """If dataset is small enough, load it all into memory
    fh = file handle
    num_of_rows = number of rows of data (i.e. molecules)
    column_headers into the hdf5 file"""
    data = np.zeros((num_of_rows,len(column_headers)))
    for key_num in range(len(column_headers)):
        key = column_headers[key_num]
        #print(key_num)
        d=fh[key]
        data[:,key_num] = d
    return data

def generate_structure_from_smiles(smiles):

    # Generate a 3D structure from smiles

    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.AddHs(mol)

    status = rdkit.AllChem.EmbedMolecule(mol)
    status = rdkit.AllChem.UFFOptimizeMolecule(mol)

    conformer = mol.GetConformer()
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)

    #atoms = get_atoms(mol)

    return coordinates

def smiles_to_persistence_diagrams(smiles):
    coords=generate_structure_from_smiles(smiles)
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
    reshaped_coords=coords[None, :, :]
    diagrams_basic = persistence.fit_transform(reshaped_coords)
    return coords, diagrams_basic

def generate_structure_from_pdb(data_dir,
                               filename):

    # Generate a 3D structure from pdb file

    mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(
        os.path.join(data_dir, filename))
    mol = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(mol)
    status = AllChem.UFFOptimizeMolecule(mol)

    conformer = mol.GetConformer()
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)

    #atoms = get_atoms(mol)

    return coordinates

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
    reshaped_coords=coords[None, :, :]
    diagrams_basic = persistence.fit_transform(reshaped_coords)
    return coords, diagrams_basic

def do_transform(transformers, dataset):
    # does deepchem transforms properly on np datasets
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    return dataset

def rotation_with_quaternion(Xx, Yy, Zz, coords, verbose=False):
    """Xx : angle around x axis
    Yy: angle around y axis
    Zz: angle around z axis
    coords: matrix Nx3 of N atoms in 3 dimensions
    Quaternions are cool"""

    # makes a quaternion
    r = R.from_euler('zyx',
    [Zz, Yy, Xx], degrees=True)
    if verbose:
        print(f'Quaternion is {r.as_quat()}')
    return r.apply(coords)

def deepchem_dataset_dictionaries():
    """returns useful intel about deepchem in this order:
    loaders = loader functions (dictionary)
    classification datasets is a list of classification datasets
    regression datasets is regression datasets
    metric types is the metric used per dataset"""

    loaders = {
          'bace_c': dc.molnet.load_bace_classification,
          'bace_r': dc.molnet.load_bace_regression,
          'bbbp': dc.molnet.load_bbbp,
          'clearance': dc.molnet.load_clearance,
          'clintox': dc.molnet.load_clintox,
          'delaney': dc.molnet.load_delaney,
          'hiv': dc.molnet.load_hiv,
          'hopv': dc.molnet.load_hopv,
          'kaggle': dc.molnet.load_kaggle,
          'kinase': dc.molnet.load_kinase,
          'lipo': dc.molnet.load_lipo,
          'muv': dc.molnet.load_muv,
          'nci': dc.molnet.load_nci,
          'pcba': dc.molnet.load_pcba,
     #     'pcba_146': dc.molnet.load_pcba_146,
     #     'pcba_2475': dc.molnet.load_pcba_2475,
          'ppb': dc.molnet.load_ppb,
          'qm7': dc.molnet.load_qm7,
    #      'qm7b': dc.molnet.load_qm7b_from_mat,
          'qm8': dc.molnet.load_qm8,
          'qm9': dc.molnet.load_qm9,
          'sampl': dc.molnet.load_sampl,
          'sider': dc.molnet.load_sider,
          'thermosol': dc.molnet.load_thermosol,
          'tox21': dc.molnet.load_tox21,
          'toxcast': dc.molnet.load_toxcast
  }

    classification_datasets = ['bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'pcba_146', 'pcba_2475', 'sider', 'tox21', 'toxcast']
    regression_datasets = ['bace_r', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo', 'nci', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl', 'thermosol']

    metric_types = {'bace_c': dc.metrics.roc_auc_score,
          'bace_r': dc.metrics.rms_score,
          'bbbp': dc.metrics.roc_auc_score,
          'clearance': dc.metrics.rms_score,
          'clintox': dc.metrics.roc_auc_score,
          'delaney': dc.metrics.mae_score,
          'hiv': dc.metrics.roc_auc_score,
          'hopv': dc.metrics.mae_score,
          'kaggle': dc.metrics.mae_score,
          'lipo': dc.metrics.mae_score,
          'muv': dc.metrics.roc_auc_score,
          'nci': dc.metrics.mae_score,
          'pcba': dc.metrics.prc_auc_score,
          'pcba_146': dc.metrics.prc_auc_score,
          'pcba_2475': dc.metrics.prc_auc_score,
          'ppb': dc.metrics.mae_score,
          'qm7': dc.metrics.mae_score,
          'qm7b': dc.metrics.mae_score,
          'qm8': dc.metrics.mae_score,
          'qm9': dc.metrics.mae_score,
          'sampl': dc.metrics.rms_score,
          'sider': dc.metrics.roc_auc_score,
          'thermosol': dc.metrics.rms_score,
          'tox21': dc.metrics.roc_auc_score,
          'toxcast': dc.metrics.roc_auc_score}
    return loaders, classification_datasets, regression_datasets, metric_types


def load_all_hdf5(fh,
                  num_of_rows,
                  column_headers):
    """If dataset is small enough, load it all into memory
    fh = file handle
    num_of_rows = number of rows of data (i.e. molecules)
    column_headers into the hdf5 file"""
    data = np.zeros((num_of_rows, len(column_headers)))
    for key_num in range(len(column_headers)):
        key = column_headers[key_num]
        # print(key_num)
        d = fh[key]
        data[:, key_num] = d
    return data


def do_transform(transformers, dataset):
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    return dataset

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
    ugh = []
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
                ugh.append(np.sqrt(egg['mean_squared_error']))
            else:
                ugh.append(egg[metric_label])
    return ugh


def topol_regression_experiment(
        dataset,
        transformers,
        Splitter_Object,
        tasks,
        metrics,
        metric_selector,
        num_repeats=5,
        num_epochs=250,
        split_fraction=[0.8, 0.1, 0.1],
        patience=15,
        metric_labels=['mean_squared_error',
                       'pearson_r2_score',
                       'mae_score',
                       'rmse']):
    """runs repeated training with topol input features
    uses default multitask regressor class
    does splitting and transforming
    returns metrics

    dataset: overall original dataset, before splitting
    transformers: deepchem transformer
    Splitter_object: deepchem splitter
    num_epochs: num epochs if not early stopping
    metric_selector: whihc one to train wiht
    split fraction: train, val, test split as decimal
    patience: for early stopping
    metric_labels: add rmse here if you want it, do not add to metrics
    metrics: list of metrics"""

    ## in function

    out = []

    frac_train = split_fraction[0]
    frac_valid = split_fraction[1]
    frac_test = split_fraction[2]
    train_scores, validate_scores, test_scores = [], [], []
    print(f'Metric selected is {metric_labels[metric_selector]}')
    # transforms datasets wooo
    dataset = do_transform(transformers, dataset)

    for i in range(num_repeats):
        # make datasets
        train_dataset, valid_dataset, test_dataset = Splitter_Object.train_valid_test_split(
            dataset=dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test)
        print(f"Training with {len(train_dataset.y)} points")
        print(f"Validation with {len(valid_dataset.y)} points")
        print(f"Testing with {len(test_dataset.y)} points")
        print(f"Total dataset size: {len(train_dataset.y) + len(valid_dataset.y) + len(test_dataset.y)}")

        # for later
        datasets = [train_dataset, valid_dataset, test_dataset]
        # actual model here
        model = dc.models.MultitaskRegressor(
            n_tasks=len(tasks),
            n_features=len(train_dataset.X[3]),
            # layer_sizes=[1000,1000,500,20],
            # dropouts=0.2,
            # learning_rate=0.001,
            residual=True)
        callback = dc.models.ValidationCallback(
            valid_dataset,
            patience,
            metrics[metric_selector])
        # fit da model
        model.fit(train_dataset, nb_epoch=num_epochs, callbacks=callback)

        # little function to calc metrics on this data
        out.append(get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers))

    pd_out = pd.DataFrame(out, columns=['tr_mse', 'tr_r2', 'tr_mae', 'tr_rmse',
                                        'val_mse', 'val_r2', 'val_mae', 'val_rmse',
                                        'te_mse', 'te_r2', 'te_mae', 'te_rmse'])
    return pd_out


def no_transform_topol_regression_experiment(
        dataset,
        Splitter_Object,
        tasks,
        metrics,
        metric_selector,
        num_repeats=5,
        num_epochs=250,
        split_fraction=[0.8, 0.1, 0.1],
        patience=15,
        metric_labels=['mean_squared_error',
                       'pearson_r2_score',
                       'mae_score',
                       'rmse']):
    """runs repeated training with topol input features
    uses default multitask regressor class
    does splitting and transforming
    returns metrics

    dataset: overall original dataset, before splitting
    transformers: deepchem transformer
    Splitter_object: deepchem splitter
    num_epochs: num epochs if not early stopping
    metric_selector: whihc one to train wiht
    split fraction: train, val, test split as decimal
    patience: for early stopping
    metric_labels: add rmse here if you want it, do not add to metrics
    metrics: list of metrics"""

    ## in function

    out = []

    frac_train = split_fraction[0]
    frac_valid = split_fraction[1]
    frac_test = split_fraction[2]
    train_scores, validate_scores, test_scores = [], [], []
    print(f'Metric selected is {metric_labels[metric_selector]}')
    for i in range(num_repeats):
        # make datasets
        train_dataset, valid_dataset, test_dataset = Splitter_Object.train_valid_test_split(
            dataset=dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test)
        print(f"Training with {len(train_dataset.y)} points")
        print(f"Validation with {len(valid_dataset.y)} points")
        print(f"Testing with {len(test_dataset.y)} points")
        print(f"Total dataset size: {len(train_dataset.y) + len(valid_dataset.y) + len(test_dataset.y)}")
        # for later
        datasets = [train_dataset, valid_dataset, test_dataset]
        # actual model here
        model = dc.models.MultitaskRegressor(
            n_tasks=len(tasks),
            n_features=len(train_dataset.X[3]),
            # layer_sizes=[1000,1000,500,20],
            # dropouts=0.2,
            # learning_rate=0.001,
            residual=True)
        callback = dc.models.ValidationCallback(
            valid_dataset,
            patience,
            metrics[metric_selector])
        # fit da model
        model.fit(train_dataset, nb_epoch=num_epochs, callbacks=callback)

        # little function to calc metrics on this data
        out.append(get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers=[]))

    pd_out = pd.DataFrame(out, columns=['tr_mse', 'tr_r2', 'tr_mae', 'tr_rmse',
                                        'val_mse', 'val_r2', 'val_mae', 'val_rmse',
                                        'te_mse', 'te_r2', 'te_mae', 'te_rmse'])
    return pd_out

def topol_classification_experiment(
        dataset,
        transformers,
        Splitter_Object,
        tasks,
        metrics,
        metric_selector,
        num_repeats=5,
        num_epochs=250,
        split_fraction=[0.8, 0.1, 0.1],
        patience=15,
        n_classes=2,
        metric_labels=['balanced_accuracy_score',
                 'prc_auc_score',
                 'roc_auc_score',
                 'f1_score']):
    """runs repeated training with topol input features
    uses default multitask regressor class
    does splitting and transforming
    returns metrics

    dataset: overall original dataset, before splitting
    transformers: deepchem transformer
    Splitter_object: deepchem splitter
    num_epochs: num epochs if not early stopping
    metric_selector: whihc one to train wiht
    split fraction: train, val, test split as decimal
    patience: for early stopping
    metric_labels: add rmse here if you want it, do not add to metrics
    metrics: list of metrics"""

    ## in function

    out = []

    frac_train = split_fraction[0]
    frac_valid = split_fraction[1]
    frac_test = split_fraction[2]
    train_scores, validate_scores, test_scores = [], [], []
    print(f'Metric selected is {metric_labels[metric_selector]}')
    # transforms datasets wooo
    dataset = do_transform(transformers, dataset)
    for i in range(num_repeats):
        # make datasets
        train_dataset, valid_dataset, test_dataset = Splitter_Object.train_valid_test_split(
            dataset=dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test)
        print(f"Training with {len(train_dataset.y)} points")
        print(f"Validation with {len(valid_dataset.y)} points")
        print(f"Testing with {len(test_dataset.y)} points")
        print(f"Total dataset size: {len(train_dataset.y) + len(valid_dataset.y) + len(test_dataset.y)}")


        # for later
        datasets = [train_dataset, valid_dataset, test_dataset]
        # actual model here
        model = dc.models.MultitaskClassifier(
            n_tasks=len(tasks),
            n_features=len(train_dataset.X[3]),
            # layer_sizes=[1000,1000,500,20],
            # dropouts=0.2,
            # learning_rate=0.001,
            n_classes=n_classes,
            residual=True)
        callback = dc.models.ValidationCallback(
            valid_dataset,
            patience,
            metrics[metric_selector])
        # fit da model
        model.fit(train_dataset, nb_epoch=num_epochs, callbacks=callback)

        # little function to calc metrics on this data
        out.append(get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers))

    pd_out = pd.DataFrame(out, columns=['tr_bal_acc', 'tr_prc_auc', 'tr_roc_auc', 'tr_f1',
                                        'val_bal_acc', 'val_prc_auc', 'val_roc_auc', 'val_f1',
                                        'te_bal_acc', 'te_prc_auc', 'te_roc_auc', 'te_f1'])
    return pd_out

def dataset_selector(
        setting,
        loader):
    """makes data inside experiment function
        setting: ECFP, CM_eig, rdkit, MACCS, 1HOT, CM
                ConvMol, Weave, Smiles2Img
        loader: loader function
    """
    print('!!!!Make this function new for each new dataset!!!!')

    if setting == 'ECFP':
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer="ECFP",
            splitter=None)
    elif setting == 'CM_eig':
        featurizer_CM_eig = dc.feat.CoulombMatrixEig(max_atoms=23)
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer=featurizer_CM_eig,
            splitter=None)
    elif setting == 'rdkit':
        featurizer_rdkit = dc.feat.RDKitDescriptors()
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer=featurizer_rdkit,
            splitter=None)

    elif setting == 'MACCS':
        featurizer = dc.feat.MACCSKeysFingerprint()
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer=featurizer,
            splitter=None)
    elif setting == '1HOT':
        featurizer = dc.feat.OneHotFeaturizer()
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer=featurizer,
            splitter=None)
    elif setting == 'CM':
        featurizer = dc.feat.CoulombMatrix(max_atoms=23)
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer=featurizer,
            splitter=None)
    elif setting == 'ConvMol':
        featurizer = dc.feat.ConvMolFeaturizer()
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer=featurizer,
            splitter=None)
    elif setting == 'Weave':
        featurizer = dc.feat.WeaveFeaturizer()
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer=featurizer,
            splitter=None)
    elif setting == 'Smiles2Img':
        # featurizer = dc.feat.SmilesToImage(
        #    img_size=250,
        #    max_len=250,
        #    res=0.5)
        tasks, datasets, transformers = loader(
            shard_size=2000,
            featurizer='smiles2img',
            splitter=None,
            img_spec='std')  # for bnw images
    else:
        print('meh no data imported')
    print
    return tasks, datasets, transformers


def model_selector(
        model_setting,
        n_tasks,
        n_features=None,
        n_classes=None,
        mode='regression'):
    """Change model settings here
    model_setting: MTR, DTNN, GraphConv
    n_tasks,
    n_features"""
    if model_setting == 'MTR':
        print('Using multitask regressor model')
        model = dc.models.MultitaskRegressor(
            n_tasks=n_tasks,
            n_features=n_features,
            # layer_sizes=[1000,1000,500,20],
            dropouts=0.2,
            # learning_rate=0.001,
            residual=True)
    elif model_setting == 'MTC':
        print('Using multitask classifier model')
        model = dc.models.MultitaskClassifier(
            n_tasks=n_tasks,
            n_features=n_features,
            # layer_sizes=[1000,1000,500,20],
            dropouts=0.2,
            # learning_rate=0.001,
            n_classes=n_classes,
            residual=True)
    elif model_setting == 'DTNN':
        print('Using DTNN model')
        model = dc.models.DTNNModel(
            n_tasks=n_tasks,
            mode=mode)
    elif model_setting == 'GraphConv':
        print('Using GraphConvMol model')
        model = dc.models.GraphConvModel(
            n_tasks=n_tasks,  # size of y, we have one output task here: finding toxicity
            weight_init_stddevs=0.01,
            bias_init_consts=0.0,
            weight_decay_penalty=0.0,
            weight_decay_penalty_type="l2",
            dropouts=0.2,
            batch_size=100,
            mode=mode,
            n_classes=n_classes
        )
    elif model_setting == 'Weave':
        print('Using Weave model')
        model = dc.models.WeaveModel(
            n_tasks=n_tasks,  # size of y, we have one output task here: finding toxicity
            # n_weave = 2,
            dropouts=0.2,
            n_classes=n_classes,
            batch_size=100,
            mode=mode
        )
    elif model_setting == 'ChemCeption':
        model = dc.models.ChemCeption(
            img_spec='std',
            n_tasks=n_tasks,
            mode=mode)
    return model


def deepchem_regression_experiment(
        metrics,
        metric_selector,
        feat_setting='ECFP',
        loader=dc.molnet.load_qm7,
        model_setting='MTR',
        dimension=1,
        num_repeats=5,
        num_epochs=250,
        split_function=dc.splits.RandomSplitter(),
        split_fraction=None,
        patience=15,
        metric_labels=['mean_squared_error',
                       'pearson_r2_score',
                       'mae_score',
                       'rmse']):
    """runs repeated training with topol input features
    uses default multitask regressor class
    does splitting and transforming
    returns metrics

    dataset: overall original dataset, before splitting
    transformers: deepchem transformer
    Splitter_object: deepchem splitter
    num_epochs: num epochs if not early stopping
    metric_selector: whihc one to train wiht
    split fraction: train, val, test split as decimal
    patience: for early stopping
    metric_labels: add rmse here if you want it, do not add to metrics
    metrics: list of metrics"""

    ## in function

    if split_fraction == None:
        split_fraction = [0.8, 0.1, 0.1]

    frac_train = split_fraction[0]
    frac_valid = split_fraction[1]
    frac_test = split_fraction[2]
    out = []
    Splitter = split_function


    train_scores, validate_scores, test_scores = [], [], []
    print(f'Metric selected is {metric_labels[metric_selector]}')
    for i in range(num_repeats):
        ################### selects different inputs ##############
        print(f'Using dataset selector setting {feat_setting}')
        tasks, dataset, transformers = dataset_selector(
            feat_setting,
            loader)

        dataset=dataset[0] # a tuple of a single dataset
        train_dataset, valid_dataset, test_dataset = Splitter.train_valid_test_split(
                        dataset=dataset,
                        frac_train=frac_train,
                        frac_valid=frac_valid,
                        frac_test=frac_test)
        datasets = (train_dataset, valid_dataset, test_dataset)
        print(f"Training with {len(train_dataset.y)} points")
        print(f"Validation with {len(valid_dataset.y)} points")
        print(f"Testing with {len(test_dataset.y)} points")
        print(f"Total dataset size: {len(train_dataset.y) + len(valid_dataset.y) + len(test_dataset.y)}")
        task_list = tasks
        ########################################################
        # actual model here
        if dimension == 1:

            model = model_selector(
                model_setting,
                n_tasks=len(tasks),
                n_features=len(train_dataset.X[3]))
        elif dimension == 2:
            model = model_selector(
                model_setting,
                n_tasks=len(tasks))
        #######################################################
        callback = dc.models.ValidationCallback(
            valid_dataset,
            patience,
            metrics[metric_selector])
        # fit da model
        model.fit(train_dataset, nb_epoch=num_epochs, callbacks=callback)

        # little function to calc metrics on this data
        out.append(get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers))

    pd_out = pd.DataFrame(out, columns=['tr_mse', 'tr_r2', 'tr_mae', 'tr_rmse',
                                        'val_mse', 'val_r2', 'val_mae', 'val_rmse',
                                        'te_mse', 'te_r2', 'te_mae', 'te_rmse'])
    return pd_out

def deepchem_classification_experiment(
        metrics,
        metric_selector,
        feat_setting='ECFP',
        loader=dc.molnet.load_qm7,
        model_setting='MTR',
        dimension=1,
        n_classes=2,
        num_repeats=5,
        num_epochs=250,
        split_function=dc.splits.RandomSplitter(),
        split_fraction=None,
        patience=15,
        metric_labels=['mean_squared_error',
                       'pearson_r2_score',
                       'mae_score',
                       'rmse']):
    """runs repeated training with topol input features
    uses default multitask regressor class
    does splitting and transforming
    returns metrics

    dataset: overall original dataset, before splitting
    transformers: deepchem transformer
    Splitter_object: deepchem splitter
    num_epochs: num epochs if not early stopping
    metric_selector: whihc one to train wiht
    split fraction: train, val, test split as decimal
    patience: for early stopping
    metric_labels: add rmse here if you want it, do not add to metrics
    metrics: list of metrics"""

    ## in function

    if split_fraction == None:
        split_fraction = [0.8, 0.1, 0.1]

    frac_train = split_fraction[0]
    frac_valid = split_fraction[1]
    frac_test = split_fraction[2]
    out = []
    Splitter = split_function


    train_scores, validate_scores, test_scores = [], [], []
    print(f'Metric selected is {metric_labels[metric_selector]}')
    for i in range(num_repeats):
        ################### selects different inputs ##############
        print(f'Using dataset selector setting {feat_setting}')
        tasks, dataset, transformers = dataset_selector(
            feat_setting,
            loader)

        dataset=dataset[0] # a tuple of a single dataset
        train_dataset, valid_dataset, test_dataset = Splitter.train_valid_test_split(
                        dataset=dataset,
                        frac_train=frac_train,
                        frac_valid=frac_valid,
                        frac_test=frac_test)
        datasets = (train_dataset, valid_dataset, test_dataset)
        print(f"Training with {len(train_dataset.y)} points")
        print(f"Validation with {len(valid_dataset.y)} points")
        print(f"Testing with {len(test_dataset.y)} points")
        print(f"Total dataset size: {len(train_dataset.y) + len(valid_dataset.y) + len(test_dataset.y)}")
        task_list = tasks
        ########################################################
        # actual model here
        if dimension == 1:

            model = model_selector(
                model_setting,
                n_tasks=len(tasks),
                n_features=len(train_dataset.X[3]),
                mode='classification',
                n_classes=2)
        elif dimension == 2:
            model = model_selector(
                model_setting,
                n_tasks=len(tasks),
                mode='classification',
                n_classes=2)
        #######################################################
        callback = dc.models.ValidationCallback(
            valid_dataset,
            patience,
            metrics[metric_selector])
        # fit da model
        model.fit(train_dataset, nb_epoch=num_epochs, callbacks=callback)

        # little function to calc metrics on this data
        out.append(get_them_metrics(
            model,
            datasets,
            metrics,
            metric_labels,
            transformers))

    pd_out = pd.DataFrame(out, columns=['tr_bal_acc', 'tr_prc_auc', 'tr_roc_auc', 'tr_f1',
                                        'val_bal_acc', 'val_prc_auc', 'val_roc_auc', 'val_f1',
                                        'te_bal_acc', 'te_prc_auc', 'te_roc_auc', 'te_f1'])
    return pd_out

def remove_specific_points(y_values, specific_points):
    """removes specific lines in the .csv file
    presumably because rdkit fails on them
    specific_points : lines to remove, 0 indexed!!!!"""
    Failures = specific_points
    new_y = np.zeros(len(y_values) - len(Failures)).reshape((-1, 1))
    count=0
    for i in range(len(y_values)):
        #print(f"i: {i}")
        if i in Failures:
            print(f"Failure: i is {i}, y_values[i] is {y_values[i]}" )
        else:
        #    print(f"count: {count}")
        #    print(dataset.y[i])
            new_y[count]=y_values[i]
            count = count + 1
    return new_y

def remove_specific_points_str(y_values, specific_points):
    """removes specific lines in the .csv file
    presumably because rdkit fails on them
    specific_points : lines to remove, 0 indexed!!!!
    does strings"""
    Failures = specific_points
    new_y = []
    count=0
    for i in range(len(y_values)):
        #print(f"i: {i}")
        if i in Failures:
            print(f"Failure: i is {i}, y_values[i] is {y_values[i]}" )
        else:
            #print(f"count: {count}")
            #print(y_values[i])
            new_y.append(y_values[i])
            count = count + 1
    return new_y

def reload_experiment_dataframes(
    list_of_filenames,
    exclusion_list=[]):
    """reloads csv files into a list of dataframes"""

    list_of_dataframes= []
    for file_no in range(len(list_of_filenames)):
        if not file_no in exclusion_list:
            this_file = list_of_filenames[file_no] + '.csv'
            print(this_file)
            list_of_dataframes.append(
                pd.read_csv(
                    os.path.join(
                        results_dir,this_file)))
    return list_of_dataframes


def method_comparison_plotter(
        list_of_data_frames,
        list_of_columns,
        color_list,
        exclusion_list=[],
        df_exclusion_list=[],
        label_list=[],
        best_con=None,
        best_con_error=None,
        best_gr=None,
        best_gr_error=None,
        x_label='',
        y_label='',
        filename='method_comparison',
        rider='all_data.png', results_dir=''):
    """Plotter
    list of data frames: list of results
    list_of_columns: which matrics to use (columsn in df)
    color_list = which colours to use
    exclusion_list: whether to miss out a df - either cos missing or unwanted
    df_exclusion_list: whether to miss out a column ONLY if not missing
    label_list: labels per dataset
    best_con: best 1D method
    best_con_error: std err
    best_gr: best graph method
    best_gr_error=None,
    x_label='',
    y_label='',
    filename='method_comparison',
    rider='all_data.png')"""

    # Best=1.92
    # pick subset of colours
    if color_list == None:
        color_list = ["#1f77f4", "#f62728",  # red blue ecfp
                      "#1f77f4", "#f62728",  # maccs
                      "#1f77f4", "#f62728",  # rdkit
                      "#88fff4", "#552200",  # brown green # cmeig
                      "#61ff33", "#ffb433",  # green orange #sm2img
                      "#61ff33", "#ffb433",  # gc
                      "#61ff33", "#ffb433",  # weave         
                      "#88fff4", "#552200",  # brown green    #cm           
                      "#5f77f4", "#f62788",  # pink blue tdaf
                      "#5f77f4", "#f62788"]  # pca-tdaf
    # get subset of labels
    if label_list == None:
        label_list = ['ECFP',
                      "MACCS",
                      "rdkit",
                      "CM_eig",
                      'Sm2Img',
                      'GC',
                      'Weave',
                      'CM',
                      "TDAF",
                      "PCA-TDAF"]
    if exclusion_list:
        color_exclusion_list = []
        for excluded in exclusion_list:
            for i in range(len(list_of_columns)):
                color_exclusion_list.append((excluded * len(list_of_columns)) + i)
        print(color_exclusion_list)
        color_list = [color_list[i] for i in range(len(color_list)) if i not in color_exclusion_list]
        print(color_list)
        # subset of labels
        egg = [label_list[i] for i in range(len(label_list)) if i not in exclusion_list]
        label_list = egg

    data = []
    for df_no in range(len(list_of_data_frames)):
        if df_no not in df_exclusion_list:
            df = list_of_data_frames[df_no]
            for col in list_of_columns:
                data.append(df[col])

    # calc and jot down the means
    print(len(data))
    means = [np.mean(x) for x in data]
    stds = [np.std(x) for x in data]
    egg = {'means': means, 'stds': stds}
    df2 = pd.DataFrame(egg)
    # saving the dataframe 
    sigh = rider.split('.')[0]
    df2.to_csv(os.path.join(results_dir, 'means_stds_in_plot_' + filename + sigh + 'csv'))

    x_positions = [x + 1 for x in range(len(means))]
    ax = plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 22})

    plt.bar(x_positions, means,
            color=color_list,
            edgecolor='k',
            linewidth='3',
            yerr=stds,
            error_kw=dict(lw=5, capsize=15, capthick=3))

    for i in range(len(means)):
        x = data[i];
        plt.plot(np.ones(len(x)) * (i + 1), x, 'o', color="#444444")

    # axes.set_ylim([0.5,4.5])
    # axes.set_xlim([0.45,4.55])
    x_tick_list = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5]

    x_tick_list = x_tick_list[:len(label_list)]
    plt.xticks(x_tick_list,
               label_list)
    if best_con:
        plt.plot([0.5, max(x_tick_list) + 1], [best_con, best_con], 'k', linewidth=4, linestyle='dashed')

    if best_con_error:
        plt.plot([0.5, max(x_tick_list) + 1], [best_con + best_con_error, best_con + best_con_error], 'grey',
                 linewidth=4, linestyle='dotted')
        plt.plot([0.5, max(x_tick_list) + 1], [best_con - best_con_error, best_con - best_con_error], 'grey',
                 linewidth=4, linestyle='dotted')

    if best_gr:
        plt.plot([0.5, max(x_tick_list) + 1], [best_con, best_con], 'g', linewidth=4, linestyle='dashed')

    if best_gr_error:
        plt.plot([0.5, max(x_tick_list) + 1], [best_con + best_con_error, best_con + best_con_error], 'darkseagreen',
                 linewidth=4, linestyle='dotted')
        plt.plot([0.5, max(x_tick_list) + 1], [best_con - best_con_error, best_con - best_con_error], 'darkseagreen',
                 linewidth=4, linestyle='dotted')

    axes = plt.gca()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(os.path.join(results_dir, filename + rider))
    return ax, means, stds