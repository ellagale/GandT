######################################################################################################################
##########################   make topological features and put them into a hdf5 file #################################
######################################################################################################################
##########################   delaney    ESOL    delaney     ESOL    ##################################################
######################################################################################################################

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
from pathlib import Path

print("TensorFlow version: " + tf.__version__)

############################## change this ###############################

dataset_name='delaney'
print(f'Using dataset {dataset_name}')
# regression datasets need to be untransformed, classification does not
is_classification = False
current_ptr = 0

make_dataset=True # whether to recalc the .csv dataset
make_hdf5 = True

do_specified_range = True
selected_range = [x for x in range(1128)] # 4201

if do_specified_range:
    current_ptr = min(selected_range)  # change this to 0 to do all of them

testing = True  # whether to only do a few molecules
if testing:
    num_of_molecules_to_do: int = len(selected_range)
else:
    num_of_molecules_to_do: int = 0  # to go into functions, 0 is an override to do all

do_PCA = False  # whether to do a PCA of the converted data

###########################################################################

# Some datasets contain molecules that cannot be featurised
Failures = []  # List of molecule IDs we expect to fail
base_dir = Path.cwd().resolve().parent  # This assumes we are in create_phf_scripts
data_dir = base_dir / "datasets" / "Delaney"
save_dir = base_dir / "output" / "converted" / f"d_{dataset_name}"
results_dir = save_dir # / f"d_{dataset_name}"
results_dir.mkdir(exist_ok=True)

test_file = dataset_name + '.csv'
out_file_name=dataset_name + '_topological_features.hdf5'
x_data_file_name = f"x_data_{dataset_name}.csv"
y_data_file_name = f"y_data_{dataset_name}.csv"
x_data_path = str(save_dir / x_data_file_name)
y_data_path = str(save_dir / y_data_file_name)



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

# Load data with no featurization

# This loads the data without doing any featurization
# also does not splitting!
tasks, datasets, transformers = loader(
    shard_size=2000,
    featurizer=h.My_Dummy_Featurizer(None),
    splitter=None) # not shuffled

dataset = datasets[0]
num_of_molecules: int = len(dataset)

batch_size = 10
if testing:
    remaining = 2
elif do_specified_range:
    remaining = len(selected_range)
else:
    remaining = num_of_molecules

#######################################################################################################################
#                       Calculate the topological features and jot them down                                          #
#######################################################################################################################


if make_dataset:
    with open(x_data_path, 'w') as f:
        with open(y_data_path, 'w') as y_fh:
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

# sys.exit(0)
##################################################################################################################
#                               load data                                                                        #
##################################################################################################################
topl_features_df = pd.read_csv(str(save_dir / x_data_file_name))
topl_targets_df = pd.read_csv(str(save_dir / y_data_file_name))

topl_delaney_all_mat=topl_features_df

delaney_df = pd.read_csv(str(data_dir / f"{dataset_name}_SMILES.csv"))  # Load the SMILES strings

with open(x_data_path, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    topl_delaney_all_list = list(csv_reader)
    print(topl_delaney_all_list)

topl_delaney_all_list = np.array(topl_delaney_all_list)

with open(y_data_path, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    targets_topl_delaney_all_list = list(csv_reader)
    # print(targets_topl_delaney_all_list)


def read_extra_data(data_dir: Path,
                    dataset_name: str,
                    Failures=[],
                    filename_rider="_SMILES.csv"):

    with open(str(data_dir / f"{dataset_name}{filename_rider}"), 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        input_SMILES_list = list(csv_reader)
        SMILES_list = h.remove_specific_points_str(y_values=input_SMILES_list, specific_points=Failures)

    return SMILES_list


SMILES_list = read_extra_data(data_dir, dataset_name, Failures,
                    filename_rider="_SMILES.csv")

names_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_names.csv")

esol_pred_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_esol_pred.csv")

min_deg_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_min_deg.csv")

MW_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_MW.csv")

num_H_bonds_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_num_H_bonds.csv")

num_rings_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_num_rings.csv")

num_rot_bonds_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_num_rot_bonds.csv")

polar_surf_list = read_extra_data(data_dir,dataset_name, Failures,
                    filename_rider="_polar_surf.csv")




################# now do the PCA ###################################################################################
if do_PCA:
    pca = PCA(n_components=num_of_topol_features)
    principalComponents_large = pca.fit_transform(topl_delaney_all_list)
####################################################################################################################
# SMILES_list = delaney_df['smiles']

mol_idx = 0

if make_hdf5:
    outfile = h5py.File(str(save_dir / out_file_name), "w")
    string_type = h5py.string_dtype(encoding='utf-8')

    num_of_molecules_override=0

    if True:

        if num_of_molecules_override == 0:
            # do all proteins woo
            num_of_molecules_to_do=len(topl_delaney_all_list)

        else:
            num_of_molecules_to_do = num_of_molecules_override
        print(f'Processing {num_of_molecules_to_do} molecules')
        ##################### set up the output datasets ################################

        # this sets up the output datasets

        molID_ds = h.create_or_recreate_dataset(outfile, "molID", (num_of_molecules_to_do), dtype=np.int64)
        SMILES_ds = h.create_or_recreate_dataset(outfile, "SMILES", (num_of_molecules_to_do), dtype=string_type)
        name_ds = h.create_or_recreate_dataset(outfile, "name", (num_of_molecules_to_do), dtype=string_type)
        esol_pred_ds = h.create_or_recreate_dataset(outfile, "esol pred", (num_of_molecules_to_do), dtype=np.float32)
        min_deg_ds = h.create_or_recreate_dataset(outfile, "min deg", (num_of_molecules_to_do), dtype=np.float32)
        MW_ds = h.create_or_recreate_dataset(outfile, "MW", (num_of_molecules_to_do), dtype=np.float32)
        num_H_bonds_ds = h.create_or_recreate_dataset(outfile, "num H bonds", (num_of_molecules_to_do), dtype=np.float32)
        num_rings_ds = h.create_or_recreate_dataset(outfile, "num rings", (num_of_molecules_to_do), dtype=np.float32)
        num_rot_bonds_ds = h.create_or_recreate_dataset(outfile, "num rotatable bonds", (num_of_molecules_to_do), dtype=np.float32)
        polar_surf_ds = h.create_or_recreate_dataset(outfile, "polar surface area", (num_of_molecules_to_do), dtype=np.float32)

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

        if do_PCA:
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
        # target_norm_ds = h.create_or_recreate_dataset(outfile, "'p_np'_norm", (num_of_molecules_to_do,), dtype=np.float32)

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

            this_num = esol_pred_list[mol_idx]
            esol_pred_ds[mol_idx] = np.array(this_num, dtype='f8')

            this_num = min_deg_list[mol_idx]
            min_deg_ds[mol_idx] = np.array(this_num, dtype='f8')

            this_num = MW_list[mol_idx]
            MW_ds[mol_idx] = np.array(this_num, dtype='f8')

            this_num = num_H_bonds_list[mol_idx]
            num_H_bonds_ds[mol_idx] = np.array(this_num, dtype='f8')

            this_num = num_rings_list[mol_idx]
            num_rings_ds[mol_idx] = np.array(this_num, dtype='f8')

            this_num = num_rot_bonds_list[mol_idx]
            num_rot_bonds_ds[mol_idx] = np.array(this_num, dtype='f8')

            this_num = polar_surf_list[mol_idx]
            polar_surf_ds[mol_idx] = np.array(this_num, dtype='f8')

                #                   toplogical features             #
            (P_pers_S_1_ds[mol_idx], P_pers_S_2_ds[mol_idx], P_pers_S_3_ds[mol_idx],
            P_no_p_1_ds[mol_idx], P_no_p_2_ds[mol_idx],P_no_p_3_ds[mol_idx],
            P_bottle_1_ds[mol_idx],P_bottle_2_ds[mol_idx],P_bottle_3_ds[mol_idx],
            P_wasser_1_ds[mol_idx],P_wasser_2_ds[mol_idx], P_wasser_3_ds[mol_idx],
            P_landsc_1_ds[mol_idx], P_landsc_2_ds[mol_idx],P_landsc_3_ds[mol_idx],
            P_pers_img_1_ds[mol_idx],P_pers_img_2_ds[mol_idx],P_pers_img_3_ds[mol_idx]
             ) = np.array(topl_delaney_all_list[mol_idx], dtype='f8')
                #                           PCs                         #
            if do_PCA:
                (PCA_1_ds[mol_idx], PCA_2_ds[mol_idx], PCA_3_ds[mol_idx], PCA_4_ds[mol_idx], PCA_5_ds[mol_idx],
                PCA_6_ds[mol_idx], PCA_7_ds[mol_idx], PCA_8_ds[mol_idx], PCA_9_ds[mol_idx], PCA_10_ds[mol_idx],
                PCA_11_ds[mol_idx], PCA_12_ds[mol_idx], PCA_13_ds[mol_idx], PCA_14_ds[mol_idx], PCA_15_ds[mol_idx],
                PCA_16_ds[mol_idx], PCA_17_ds[mol_idx], PCA_18_ds[mol_idx])=principalComponents_large[mol_idx]

            # targets
            try:
                # csvreader is being stupid and reading a 1 as 1.0.
                target_ds[mol_idx] = np.array([int(float(x)) for x in targets_topl_delaney_all_list[mol_idx]])
            except TypeError as e:
                raise (e)
            #target_norm_ds[mol_idx] = np.array(targets_topl_delaney_all_list[mol_idx], dtype='f8')

    outfile.close()

sys.exit(0)





