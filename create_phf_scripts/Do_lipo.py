######################################################################################################################
##########################   make topological features and put them into a hdf5 file #################################
######################################################################################################################
##########################   lipo   lipo    lipo    lipo    lipo    lipo    lipo     #################################
######################################################################################################################

import sys
sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

import deepchem as dc

import tensorflow as tf
import os
import sys
import rdkit
import h5py
from src import helper_functions as h

from csv import reader
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

print("TensorFlow version: " + tf.__version__)

############################## change this ###############################

dataset_name='lipo'
print(f'Using dataset {dataset_name}')
# regression datasets need to be untransformed, classification does not
is_classification = False
current_ptr = 0

make_dataset=True # whether to recalc the .csv dataset
make_hdf5 = True

do_specified_range = True
selected_range = [x for x in range(4198)] # 4201
num_of_molecules_to_do = len(selected_range)
if do_specified_range:
    current_ptr = min(selected_range)
  ## change this to 0 to do all of them
testing = False # whether to only do a few molecules
if testing:
    num_of_molecules_to_do = len(selected_range)
else:
    num_of_molecules_to_do: int = 0  # to go into functions, 0 is an override to do all
#num_of_molecules_to_do = len(selected_range)
###########################################################################

#lipo has some molecules that cannot be featurised :(
Failures = [3592] #

save_dir=r'C:\Users\eg16993\OneDrive - University of Bristol\Documents\Datasets\topol_datasets'
data_dir=r'C:\Users\eg16993\OneDrive - University of Bristol\Documents\Datasets'
results_dir=r"C:\Users\eg16993\OneDrive - University of Bristol\Documents\Results\graphs_and_topology\d_" + dataset_name

test_file=dataset_name + '.csv'
out_file_name=dataset_name + '_topological_features.hdf5'
x_data_file_name = 'x_data_' + dataset_name + '.csv'
y_data_file_name = 'y_data_' + dataset_name + '.csv'



loaders, classification_datasets, regression_datasets, metric_types = h.deepchem_dataset_dictionaries()
loader = loaders[dataset_name]

print(f"DeepChem version: {dc.__version__}")
#
feature_name_list = ['pers_S_1', 'pers_S_2', 'pers_S_3',
                    'no_p_1', 'no_p_2', 'no_p_3',
                    'bottle_1', 'bottle_2', 'bottle_3',
                    'wasser_1', 'wasser_2', 'wasser_3',
                    'landsc_1', 'landsc_2', 'landsc_3',
                    'pers_img_1', 'pers_img_2', 'pers_img_3']
num_of_topol_features = len(feature_name_list)

#### Load data with no featurization

# This loads the data without doing any featurization
# also does not splitting!
tasks, datasets, transformers = loader(
    shard_size=2000,
    featurizer=h.My_Dummy_Featurizer(None),
    splitter=None) # not shuffled

dataset = datasets[0]
num_of_molecules: int = len(dataset)

# RUN THIS
#make_dataset = False



batch_size = 10
if testing:
    remaining = 9
elif do_specified_range:
    remaining = len(selected_range)
else:
    remaining = num_of_molecules

#######################################################################################################################
#                       Calculate the topological features and jot them down                                          #
#######################################################################################################################

#make_dataset = True
if make_dataset:
    with open(os.path.join(save_dir,x_data_file_name), 'w') as f:
        with open(os.path.join(save_dir, y_data_file_name), 'w') as y_fh:
            # train
            if not testing:
                remaining = len(dataset)
            Failures = h.temp_write_topol_data(
                f,
                remaining=remaining,
                current_ptr=current_ptr,
                my_dataset=dataset,
                num_of_topol_features=num_of_topol_features,
                do_specified_range=do_specified_range,
                batch_size=batch_size,
                data_dir=data_dir,
                file_type='smiles',
                skip_molecules=Failures
            )
            print('#################################################################')
            print(f'Failures in {dataset_name} are: {Failures}')
            print('#################################################################')
            if not is_classification:
                untransformed_train_y = transformers[0].untransform(dataset.y)
            else:
                untransformed_train_y = dataset.y
            final_y = h.remove_specific_points(
                y_values=untransformed_train_y,
                specific_points=Failures)
            h.copy_targets_into_csv(
                y_fh=y_fh,
                my_dataset=untransformed_train_y
            )

    f.close()

#sys.exit(0)
##################################################################################################################
#                               load data                                                                        #
##################################################################################################################
topl_features_df = pd.read_csv(os.path.join(save_dir,x_data_file_name))
topl_targets_df = pd.read_csv(os.path.join(save_dir,y_data_file_name))

topl_lipo_all_mat=topl_features_df

lipo_df=pd.read_csv(os.path.join(save_dir, dataset_name + '_SMILES.csv')) # has smiles strings

with open(os.path.join(save_dir,x_data_file_name), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    topl_lipo_all_list = list(csv_reader)
    print(topl_lipo_all_list)

topl_lipo_all_list = np.array(topl_lipo_all_list)

with open(os.path.join(save_dir,y_data_file_name), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    targets_topl_lipo_all_list = list(csv_reader)
    #print(targets_topl_lipo_all_list)

with open(os.path.join(save_dir,dataset_name + '_SMILES.csv'), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    input_SMILES_list = list(csv_reader)
    SMILES_list=h.remove_specific_points_str(y_values=input_SMILES_list, specific_points=Failures)


with open(os.path.join(save_dir,dataset_name + '_names.csv'), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    input_names_list = list(csv_reader)
    names_list=h.remove_specific_points_str(y_values=input_names_list, specific_points=Failures)

################# now do the PCA ###################################################################################
#pca = PCA(n_components=num_of_topol_features)
pca = PCA(n_components=num_of_topol_features)
principalComponents_large = pca.fit_transform(topl_lipo_all_list)
####################################################################################################################
#SMILES_list = lipo_df['smiles']

mol_idx = 0

if make_hdf5:
    outfile = h5py.File(os.path.join(save_dir,out_file_name),"w")
    string_type = h5py.string_dtype(encoding='utf-8')

    num_of_molecules_override=0

    if True:

        if num_of_molecules_override == 0:
            # do all proteins woo
            num_of_molecules_to_do=len(topl_lipo_all_list)

        else:
            num_of_molecules_to_do = num_of_molecules_override
        print(f'Processing {num_of_molecules_to_do} molecules')
        ##################### set up the output datasets ################################

        ## this sets up the output datasets

        molID_ds = h.create_or_recreate_dataset(outfile, "molID", (num_of_molecules_to_do), dtype=np.int64)
        SMILES_ds = h.create_or_recreate_dataset(outfile, "SMILES", (num_of_molecules_to_do), dtype=string_type)
        name_ds = h.create_or_recreate_dataset(outfile, "name", (num_of_molecules_to_do), dtype=string_type)
            #                                       ##### topological data ###                              #
        ###### proteins #####
        #      Persistence entropy
        P_pers_S_1_ds = h.create_or_recreate_dataset(outfile, "pers_S_1", (num_of_molecules_to_do), dtype=np.float32)
        P_pers_S_2_ds = h.create_or_recreate_dataset(outfile, "pers_S_2", (num_of_molecules_to_do), dtype=np.float32)
        P_pers_S_3_ds = h.create_or_recreate_dataset(outfile, "pers_S_3", (num_of_molecules_to_do), dtype=np.float32)
        #      No. of points
        P_no_p_1_ds = h.create_or_recreate_dataset(outfile, "no_p_1", (num_of_molecules_to_do), dtype=np.float32)
        P_no_p_2_ds = h.create_or_recreate_dataset(outfile, "no_p_2", (num_of_molecules_to_do), dtype=np.float32)
        P_no_p_3_ds = h.create_or_recreate_dataset(outfile, "no_p_3", (num_of_molecules_to_do), dtype=np.float32)
        #      Bottleneck
        P_bottle_1_ds = h.create_or_recreate_dataset(outfile, "bottle_1", (num_of_molecules_to_do), dtype=np.float32)
        P_bottle_2_ds = h.create_or_recreate_dataset(outfile, "bottle_2", (num_of_molecules_to_do), dtype=np.float32)
        P_bottle_3_ds = h.create_or_recreate_dataset(outfile, "bottle_3", (num_of_molecules_to_do), dtype=np.float32)
        #      Wasserstein
        P_wasser_1_ds = h.create_or_recreate_dataset(outfile, "wasser_1", (num_of_molecules_to_do), dtype=np.float32)
        P_wasser_2_ds = h.create_or_recreate_dataset(outfile, "wasser_2", (num_of_molecules_to_do), dtype=np.float32)
        P_wasser_3_ds = h.create_or_recreate_dataset(outfile, "wasser_3", (num_of_molecules_to_do), dtype=np.float32)
        #      landscape
        P_landsc_1_ds = h.create_or_recreate_dataset(outfile, "landsc_1", (num_of_molecules_to_do), dtype=np.float32)
        P_landsc_2_ds = h.create_or_recreate_dataset(outfile, "landsc_2", (num_of_molecules_to_do), dtype=np.float32)
        P_landsc_3_ds = h.create_or_recreate_dataset(outfile, "landsc_3", (num_of_molecules_to_do), dtype=np.float32)
        #      persistence image
        P_pers_img_1_ds = h.create_or_recreate_dataset(outfile, "pers_img_1", (num_of_molecules_to_do), dtype=np.float32)
        P_pers_img_2_ds = h.create_or_recreate_dataset(outfile, "pers_img_2", (num_of_molecules_to_do), dtype=np.float32)
        P_pers_img_3_ds = h.create_or_recreate_dataset(outfile, "pers_img_3", (num_of_molecules_to_do), dtype=np.float32)

            #                                       PCs                                                 #
        PCA_1_ds = h.create_or_recreate_dataset(outfile, "PCA_1", (num_of_molecules_to_do), dtype=np.float32)
        PCA_2_ds = h.create_or_recreate_dataset(outfile, "PCA_2", (num_of_molecules_to_do), dtype=np.float32)
        PCA_3_ds = h.create_or_recreate_dataset(outfile, "PCA_3", (num_of_molecules_to_do), dtype=np.float32)
        PCA_4_ds = h.create_or_recreate_dataset(outfile, "PCA_4", (num_of_molecules_to_do), dtype=np.float32)
        PCA_5_ds = h.create_or_recreate_dataset(outfile, "PCA_5", (num_of_molecules_to_do), dtype=np.float32)
        PCA_6_ds = h.create_or_recreate_dataset(outfile, "PCA_6", (num_of_molecules_to_do), dtype=np.float32)
        PCA_7_ds = h.create_or_recreate_dataset(outfile, "PCA_7", (num_of_molecules_to_do), dtype=np.float32)
        PCA_8_ds = h.create_or_recreate_dataset(outfile, "PCA_8", (num_of_molecules_to_do), dtype=np.float32)
        PCA_9_ds = h.create_or_recreate_dataset(outfile, "PCA_9", (num_of_molecules_to_do), dtype=np.float32)
        PCA_10_ds = h.create_or_recreate_dataset(outfile, "PCA_10", (num_of_molecules_to_do), dtype=np.float32)
        PCA_11_ds = h.create_or_recreate_dataset(outfile, "PCA_11", (num_of_molecules_to_do), dtype=np.float32)
        PCA_12_ds = h.create_or_recreate_dataset(outfile, "PCA_12", (num_of_molecules_to_do), dtype=np.float32)
        PCA_13_ds = h.create_or_recreate_dataset(outfile, "PCA_13", (num_of_molecules_to_do), dtype=np.float32)
        PCA_14_ds = h.create_or_recreate_dataset(outfile, "PCA_14", (num_of_molecules_to_do), dtype=np.float32)
        PCA_15_ds = h.create_or_recreate_dataset(outfile, "PCA_15", (num_of_molecules_to_do), dtype=np.float32)
        PCA_16_ds = h.create_or_recreate_dataset(outfile, "PCA_16", (num_of_molecules_to_do), dtype=np.float32)
        PCA_17_ds = h.create_or_recreate_dataset(outfile, "PCA_17", (num_of_molecules_to_do), dtype=np.float32)
        PCA_18_ds = h.create_or_recreate_dataset(outfile, "PCA_18", (num_of_molecules_to_do), dtype=np.float32)
            # # #                           targets                                         # # #
        target_ds = h.create_or_recreate_dataset(outfile, "p_np", (num_of_molecules_to_do), dtype=np.int8)
        #target_norm_ds = h.create_or_recreate_dataset(outfile, "'p_np'_norm", (num_of_molecules_to_do,), dtype=np.float32)

        for mol_idx in range(num_of_molecules_to_do):
            if mol_idx % 50 == 0:
                print('Got to Molecule no. ', mol_idx, SMILES_list[mol_idx])
            molID_ds[mol_idx] = mol_idx
            # get the current PDB code
            #current_code=PDB_List[mol_idx]
            # get the rows in the dataframes
            #current_index_row=df_index.loc[df_index['PDB_code']==current_code]
            #print(current_code)
            #print(current_index_row)
            #current_data_row=df_data.loc[df_data['PDB_code']==current_code]
            #print(current_data_row)

            smiles_string = SMILES_list[mol_idx]
            SMILES_ds[mol_idx] = np.array(smiles_string, dtype=string_type)

            name_string = names_list[mol_idx]
            name_ds[mol_idx] = np.array(name_string, dtype=string_type)

                #                   toplogical features             #
            (P_pers_S_1_ds[mol_idx], P_pers_S_2_ds[mol_idx], P_pers_S_3_ds[mol_idx],
            P_no_p_1_ds[mol_idx], P_no_p_2_ds[mol_idx],P_no_p_3_ds[mol_idx],
            P_bottle_1_ds[mol_idx],P_bottle_2_ds[mol_idx],P_bottle_3_ds[mol_idx],
            P_wasser_1_ds[mol_idx],P_wasser_2_ds[mol_idx], P_wasser_3_ds[mol_idx],
            P_landsc_1_ds[mol_idx], P_landsc_2_ds[mol_idx],P_landsc_3_ds[mol_idx],
            P_pers_img_1_ds[mol_idx],P_pers_img_2_ds[mol_idx],P_pers_img_3_ds[mol_idx]
             ) = np.array(topl_lipo_all_list[mol_idx], dtype='f8')
                #                           PCs                         #
            (PCA_1_ds[mol_idx], PCA_2_ds[mol_idx], PCA_3_ds[mol_idx], PCA_4_ds[mol_idx], PCA_5_ds[mol_idx],
            PCA_6_ds[mol_idx], PCA_7_ds[mol_idx], PCA_8_ds[mol_idx], PCA_9_ds[mol_idx], PCA_10_ds[mol_idx],
            PCA_11_ds[mol_idx], PCA_12_ds[mol_idx], PCA_13_ds[mol_idx], PCA_14_ds[mol_idx], PCA_15_ds[mol_idx],
            PCA_16_ds[mol_idx], PCA_17_ds[mol_idx], PCA_18_ds[mol_idx])=principalComponents_large[mol_idx]

            # targets
            try:
                # csvreader is being stupid and reading a 1 as 1.0.
                target_ds[mol_idx] = np.array([int(float(x)) for x in targets_topl_lipo_all_list[mol_idx]])
            except TypeError as e:
                raise (e)
            #target_norm_ds[mol_idx] = np.array(targets_topl_lipo_all_list[mol_idx], dtype='f8')

    outfile.close()

sys.exit(0)





