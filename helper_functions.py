# ######################################################################################
#           Helper functions for graphs and topology in deepchem                       #
########################################################################################
import os
import pandas as pd
import rdkit
import numpy as np
# topology stuff
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import NumberOfPoints
from gtda.diagrams import Amplitude
from sklearn.pipeline import make_union, Pipeline


def coord_getter(data_dir, test_file, test_pdb_code, setting='pdb'):
    """New quick function to grab the coordinates from the files
    Doesn't do all the conformer and rotation stuff that the functions in
    projection does"""
    test_file_location=os.path.join(data_dir, test_pdb_code, test_file)
    if setting=='pdb':
        mol_orig=rdkit.Chem.rdmolfiles.MolFromPDBFile(test_file_location)
    elif setting=='mol2':
        mol_orig = rdkit.Chem.rdmolfiles.MolFromMol2File(
            test_file_location,
            cleanupSubstructures=False,
            sanitize=False)
    egg=mol_orig.GetConformer()
    rdkit.Chem.rdMolTransforms.CanonicalizeConformer(egg)
    coords=egg.GetPositions()
    return mol_orig, coords

def coord_getter_2(file_location, setting='pdb'):
    print(file_location)
    if setting=='pdb':
        mol_orig=rdkit.Chem.rdmolfiles.MolFromPDBFile(file_location, sanitize=False)
        #rdkit.Chem.rdMolTransforms.CanonicalizeConformer(self.conformer)
    elif setting=='mol2':
        mol_orig = rdkit.Chem.rdmolfiles.MolFromMol2File(
            file_location,
            cleanupSubstructures=False,
            sanitize=False)
    egg=mol_orig.GetConformer()
    rdkit.Chem.rdMolTransforms.CanonicalizeConformer(egg)
    coords=egg.GetPositions()
    return mol_orig, coords

def read_in_PDBBind_data(
    data_dir,
    name_file_name="INDEX_core_name.2013",
    data_file_name="INDEX_core_data.2013",
    cluster_file_name = "INDEX_core_cluster.2013"):
#if True:

    """script to read into the data for PDBBind"""

    # reads in the name data
    if not name_file_name == '':
        ## This reads in the pdb codes for each protein in the core dataset
        fh = open(os.path.join(data_dir, name_file_name),'r')
        c = 0
        column_list_name=["PDB_code", "release_year", "EC_number", "protein_name"]
        lines=[]
        for line in fh.readlines():
            if not line.startswith('#'):
                if c ==0:
                    words=[ x for x in line.split(' ') if not x == '']
                    last_word = words[3:]
                    last_word[-1] = last_word[-1].strip()
                    last_word = '_'.join(last_word)
                    new_line=[words[0],words[1],words[2], last_word]
                    #print(new_line)
                    lines.append(new_line)
                    #df_line = pd.DataFrame([new_line],columns=column_list)
                    #print(df_line)
                    #df_index_core.append(df_line)
                    #labels = line

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
        fh = open(os.path.join(data_dir, data_file_name),'r')
        c = 0
        column_list_data=["PDB_code",
                     "resolution",
                     "release_year",
                     "-logKd/Ki",
                     "Kd/Ki",
                     "reference",
                     "ligand name"]
        lines=[]
        for line in fh.readlines():
            if not line.startswith('#'):
                if c ==0:
                    words=[ x for x in line.split(' ') if not x == '']
                    # Not actually the last word!
                    last_word = words[5:-1]
                    last_word[-1] = last_word[-1].strip()
                    last_word = '_'.join(last_word)

                    new_line=[words[0],words[1],words[2],words[3],words[4], last_word, words[-1][1:-2]]
                    #print(new_line)
                    lines.append(new_line)
                    #df_line = pd.DataFrame([new_line],columns=column_list)
                    #print(df_line)
                    #df_index_core.append(df_line)
                    #labels = line

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
        fh = open(os.path.join(data_dir, cluster_file_name),'r')
        c = 0
        column_list_cluster=['PDB_code',
                     'resolution',
                     'release_year',
                     '-logKd/Ki',
                     'original_Kd/Ki',
                     'cluster ID']
        lines=[]
        for line in fh.readlines():
            if not line.startswith('#'):
                if c ==0:
                    words=[ x for x in line.split(' ') if not x == '']
                    last_word = words[5:]
                    last_word[-1] = last_word[-1].strip()
                    last_word = '_'.join(last_word)
                    new_line=[words[0],words[1],words[2],words[3],words[4], last_word]
                    #print(new_line)
                    lines.append(new_line)
                    #df_line = pd.DataFrame([new_line],columns=column_list)
                    #print(df_line)
                    #df_index_core.append(df_line)
                    #labels = line

        fh.close()
        df_cluster_core = pd.DataFrame(lines, columns=column_list_cluster)
        df_cluster_core.head()
    else:
        df_cluster_core = ''
    print('Sample of cluster file:')
    print(df_cluster_core.head())
    return df_index_core, df_data_core, df_cluster_core


def make_topological_features_for_PDBBind(df_cluster_core,
                                          PDB_or_mol2='mol2',
                                          verbose=True,
                                          Num_of_proteins=0,
                                          Num_of_features=3,
                                          data_dir=''):
    """makes topological features from pdb and mol2 files in PDBBind
    df_cluster_core = dataframe of the cluster dataset
    PDB_or_mol2='mol2': pdb fro proteins, mol2 for ligands
    Num_of_proteins=0 nume of proteins to do, 0 is all
    Num_of_features: 3 for persistence entropy only, 9 for everything
    """
    if Num_of_proteins == 0:
        Num_of_proteins = len(df_cluster_core)
    point_ptr = -1
    # PDB_or_mol2='pdb'
    pdb_list = df_cluster_core['PDB_code']
    if PDB_or_mol2 == 'mol2':
        input_file_end_name = 'ligand'
    elif PDB_or_mol2 == 'pdb':
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

    for mol_idx in range(Num_of_proteins):
        if mol_idx % 50 == 0:
            print('Got to Molecule no. ', mol_idx)
        ### load da data
        file_location = os.path.join(data_dir,
                                     pdb_list[mol_idx],
                                     pdb_list[mol_idx] + '_' + input_file_end_name + '.' + PDB_or_mol2)
        if PDB_or_mol2 == 'mol2':
            mol, coords = coord_getter_2(file_location, setting='mol2')
        elif PDB_or_mol2 == 'pdb':
            mol, coords = coord_getter_2(file_location, setting='pdb')

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
        print(X_basic)

    topol_feat_mat = np.array(topol_feat_list)

    return topol_feat_list, topol_feat_mat


def create_and_merge_PDBBind_topol_features(df_cluster_core,
                                            Num_of_proteins=5,
                                            Num_of_features=18,
                                            data_dir='',
                                            verbose=False):
    """merges topological features for ligand nad protein into the same dataset
    i.e. each row has x protein features and y ligand features"""
    # grab ligands
    print('Doing the proteins...')
    topol_feat_list_protein, topol_feat_mat_protein = make_topological_features_for_PDBBind(df_cluster_core,
                                                                                          PDB_or_mol2='pdb',
                                                                                          verbose=verbose,
                                                                                          Num_of_proteins=Num_of_proteins,
                                                                                          Num_of_features=Num_of_features,
                                                                                          data_dir=data_dir)
    # do proteins
    print('Doing the ligands')
    topol_feat_list_ligand, topol_feat_mat_ligand = make_topological_features_for_PDBBind(df_cluster_core,
                                                                                           PDB_or_mol2='mol2',
                                                                                           verbose=verbose,
                                                                                           Num_of_proteins=Num_of_proteins,
                                                                                           Num_of_features=Num_of_features,
                                                                                           data_dir=data_dir)
    # list version of data
    topl_PDB_all_core = [topol_feat_list_protein[i] + topol_feat_list_ligand[i] for i in
                         range(len(topol_feat_list_ligand))]

    # numpy version of data
    topl_PDB_all_core_mat = np.array(topl_PDB_all_core)
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
            test_set_size = 0.1*num_of_proteins
            print('test set has {} items'.format(test_set_size))
    if validate_set_size == None:
        if verbose:
            print('Assuming a 10% validation set')
            validate_set_size = 0.1*num_of_proteins
            print('test set has {} items'.format(validate_set_size))

    test_data_indices = np.random.choice(num_of_proteins, test_set_size, replace=False)
    validate_data_indices = np.random.choice(num_of_proteins, validate_set_size, replace=False)
    not_train_indices = np.concatenate((test_data_indices, validate_data_indices))
    # print(test_data_indices)
    test_X_data = [X_data[i] for i in test_data_indices]
    validate_X_data = [X_data[i] for i in validate_data_indices]
    train_X_data = [X_data[i] for i in range(num_of_proteins) if i not in not_train_indices]
    y = y_data[:num_of_proteins]
    test_y_data = [float(y[i]) for i in test_data_indices]
    validate_y_data = [float(y[i]) for i in validate_data_indices]
    train_y_data = [float(y[i]) for i in range(num_of_proteins) if i not in not_train_indices]
    if verbose:
        print('Test set indices:')
        print(test_data_indices)
        print('Validation set indices:')
        print(validate_data_indices)
    return (train_X_data, train_y_data, test_X_data, test_y_data, validate_X_data, validate_y_data)
