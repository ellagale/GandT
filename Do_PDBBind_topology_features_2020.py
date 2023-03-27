######################################################################################################################
##########################   make topological features and put them into a hdf5 file #################################
######################################################################################################################

# TODO : tidy this file up and check it, it is a bit messay I think 



# fixc this at some point
import sys
sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

import tensorflow as tf
import os
import sys
import rdkit
import h5py
import helper_functions as h

import numpy as np


print("TensorFlow version: " + tf.__version__)

# fixc this at some point
sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")


# CHANGE THIS IF NECESSARY
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
name_file_name= "INDEX_refined_name.2013",
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
out_file_name="PDBBind_core_topological_features.hdf5"
# Open hdf5 file, calc basic details
#outfile = out_file_name
print(out_file_name)
fh = h5py.File(os.path.join(save_dir,out_file_name), 'r+')
num_of_rows, num_of_molecules = h.basic_info_hdf5_dataset(fh, label='molID')
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
X_data = X_data_large
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

make_dataset=False
if make_dataset:
    Num_of_proteins = 0## change this to 0 to do all of them
    topl_PDB_all_core_large, topl_PDB_all_core_mat_large=h.create_and_merge_PDBBind_topol_features(
                                df_index_core,
                                verbose=False,
                                Num_of_proteins=Num_of_proteins,
                                Num_of_features=18,
                                data_dir=data_dir)

    #topl_PDB_all_core_small, topl_PDB_all_core_mat_small=h.create_and_merge_PDBBind_topol_features(
    #                            df_index_core,
    #                            verbose=False,
    #                            Num_of_proteins=Num_of_proteins,
    #                            Num_of_features=3,
    #                            data_dir=data_dir)

    topl_PDB_all_core_small = [x[0:3] for x in topl_PDB_all_core_large]
    #topl_PDB_all_core_small
    topl_PDB_all_core_mat_small = np.array(topl_PDB_all_core_small)

# RUN THIS
make_dataset = False
do_specified_range = True
selected_range = [x for x in range(7, 10)]
Num_of_proteins = 0  ## change this to 0 to do all of them

from pathlib import Path
import numpy as np
import os
import csv

current_ptr = 0
batch_size = 100
remaining = len(df_index_refined)

if make_dataset:
    with open('output.csv', 'w') as f:
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            this_range = list(range(current_ptr, current_ptr + this_batch))

            out_list, y_new = h.create_and_merge_PDBBind_topol_features(
                df_index_refined,
                verbose=False,
                Num_of_proteins=Num_of_proteins,
                Num_of_features=18,
                data_dir=data_dir,
                do_specified_range=True,
                selected_range=this_range)

            for _list in out_list:
                row = ','.join([str(i) for i in _list])
                f.write(row + '\n')
            remaining -= this_batch
            current_ptr += this_batch

    f.close()

pca = PCA(n_components=36)
principalComponents_core = pca.fit_transform(topl_PDB_all_core_large)

if make_dataset:
    outfile = h5py.File(os.path.join(save_dir,out_file_name),"w")
    string_type = h5py.string_dtype(encoding='utf-8')

    PDB_List=df_index_core['PDB_code']
    df_index = df_index_core
    df_data = df_data_core
    df_cluster = df_cluster_core
    topl_PDB_large = topl_PDB_all_core_large

    num_of_proteins_override=0#21
    do_cluster=True

    if True:

        if num_of_proteins_override == 0:
            # do all proteins woo
            Num_of_proteins= len(PDB_List)
        else:
            Num_of_proteins = num_of_proteins_override

        ##################### set up the output datasets ################################

        ## this sets up the output datasets
        molID_ds = h.h.create_or_recreate_dataset(outfile, "molID", (Num_of_proteins,), dtype=np.int8)
        ###### topological data ####
        ###### proteins #####
        #      Persistence entropy
        P_pers_S_1_ds = h.create_or_recreate_dataset(outfile, "P_pers_S_1", (Num_of_proteins,), dtype=np.float32)
        P_pers_S_2_ds = h.create_or_recreate_dataset(outfile, "P_pers_S_2", (Num_of_proteins,), dtype=np.float32)
        P_pers_S_3_ds = h.create_or_recreate_dataset(outfile, "P_pers_S_3", (Num_of_proteins,), dtype=np.float32)
        #      No. of points
        P_no_p_1_ds = h.create_or_recreate_dataset(outfile, "P_no_p_1", (Num_of_proteins,), dtype=np.float32)
        P_no_p_2_ds = h.create_or_recreate_dataset(outfile, "P_no_p_2", (Num_of_proteins,), dtype=np.float32)
        P_no_p_3_ds = h.create_or_recreate_dataset(outfile, "P_no_p_3", (Num_of_proteins,), dtype=np.float32)
        #      Bottleneck
        P_bottle_1_ds = h.create_or_recreate_dataset(outfile, "P_bottle_1", (Num_of_proteins,), dtype=np.float32)
        P_bottle_2_ds = h.create_or_recreate_dataset(outfile, "P_bottle_2", (Num_of_proteins,), dtype=np.float32)
        P_bottle_3_ds = h.create_or_recreate_dataset(outfile, "P_bottle_3", (Num_of_proteins,), dtype=np.float32)
        #      Wasserstein
        P_wasser_1_ds = h.create_or_recreate_dataset(outfile, "P_wasser_1", (Num_of_proteins,), dtype=np.float32)
        P_wasser_2_ds = h.create_or_recreate_dataset(outfile, "P_wasser_2", (Num_of_proteins,), dtype=np.float32)
        P_wasser_3_ds = h.create_or_recreate_dataset(outfile, "P_wasser_3", (Num_of_proteins,), dtype=np.float32)
        #      landscape
        P_landsc_1_ds = h.create_or_recreate_dataset(outfile, "P_landsc_1", (Num_of_proteins,), dtype=np.float32)
        P_landsc_2_ds = h.create_or_recreate_dataset(outfile, "P_landsc_2", (Num_of_proteins,), dtype=np.float32)
        P_landsc_3_ds = h.create_or_recreate_dataset(outfile, "P_landsc_3", (Num_of_proteins,), dtype=np.float32)
        #      persistence image
        P_pers_img_1_ds = h.create_or_recreate_dataset(outfile, "P_pers_img_1", (Num_of_proteins,), dtype=np.float32)
        P_pers_img_2_ds = h.create_or_recreate_dataset(outfile, "P_pers_img_2", (Num_of_proteins,), dtype=np.float32)
        P_pers_img_3_ds = h.create_or_recreate_dataset(outfile, "P_pers_img_3", (Num_of_proteins,), dtype=np.float32)
        #### ligands ####
        #      Persistence entropy
        L_pers_S_1_ds = h.create_or_recreate_dataset(outfile, "L_pers_S_1", (Num_of_proteins,), dtype=np.float32)
        L_pers_S_2_ds = h.create_or_recreate_dataset(outfile, "L_pers_S_2", (Num_of_proteins,), dtype=np.float32)
        L_pers_S_3_ds = h.create_or_recreate_dataset(outfile, "L_pers_S_3", (Num_of_proteins,), dtype=np.float32)
        #      No. of points
        L_no_p_1_ds = h.create_or_recreate_dataset(outfile, "L_no_p_1", (Num_of_proteins,), dtype=np.float32)
        L_no_p_2_ds = h.create_or_recreate_dataset(outfile, "L_no_p_2", (Num_of_proteins,), dtype=np.float32)
        L_no_p_3_ds = h.create_or_recreate_dataset(outfile, "L_no_p_3", (Num_of_proteins,), dtype=np.float32)
        #      Bottleneck
        L_bottle_1_ds = h.create_or_recreate_dataset(outfile, "L_bottle_1", (Num_of_proteins,), dtype=np.float32)
        L_bottle_2_ds = h.create_or_recreate_dataset(outfile, "L_bottle_2", (Num_of_proteins,), dtype=np.float32)
        L_bottle_3_ds = h.create_or_recreate_dataset(outfile, "L_bottle_3", (Num_of_proteins,), dtype=np.float32)
        #      Wasserstein
        L_wasser_1_ds = h.create_or_recreate_dataset(outfile, "L_wasser_1", (Num_of_proteins,), dtype=np.float32)
        L_wasser_2_ds = h.create_or_recreate_dataset(outfile, "L_wasser_2", (Num_of_proteins,), dtype=np.float32)
        L_wasser_3_ds = h.create_or_recreate_dataset(outfile, "L_wasser_3", (Num_of_proteins,), dtype=np.float32)
        #      landscape
        L_landsc_1_ds = h.create_or_recreate_dataset(outfile, "L_landsc_1", (Num_of_proteins,), dtype=np.float32)
        L_landsc_2_ds = h.create_or_recreate_dataset(outfile, "L_landsc_2", (Num_of_proteins,), dtype=np.float32)
        L_landsc_3_ds = h.create_or_recreate_dataset(outfile, "L_landsc_3", (Num_of_proteins,), dtype=np.float32)
        #      persistence image
        L_pers_img_1_ds = h.create_or_recreate_dataset(outfile, "L_pers_img_1", (Num_of_proteins,), dtype=np.float32)
        L_pers_img_2_ds = h.create_or_recreate_dataset(outfile, "L_pers_img_2", (Num_of_proteins,), dtype=np.float32)
        L_pers_img_3_ds = h.create_or_recreate_dataset(outfile, "L_pers_img_3", (Num_of_proteins,), dtype=np.float32)

        PCA_1_ds = h.create_or_recreate_dataset(outfile, "PCA_1", (Num_of_proteins,), dtype=np.float32)
        PCA_2_ds = h.create_or_recreate_dataset(outfile, "PCA_2", (Num_of_proteins,), dtype=np.float32)
        PCA_3_ds = h.create_or_recreate_dataset(outfile, "PCA_3", (Num_of_proteins,), dtype=np.float32)
        PCA_4_ds = h.create_or_recreate_dataset(outfile, "PCA_4", (Num_of_proteins,), dtype=np.float32)
        PCA_5_ds = h.create_or_recreate_dataset(outfile, "PCA_5", (Num_of_proteins,), dtype=np.float32)
        PCA_6_ds = h.create_or_recreate_dataset(outfile, "PCA_6", (Num_of_proteins,), dtype=np.float32)
        PCA_7_ds = h.create_or_recreate_dataset(outfile, "PCA_7", (Num_of_proteins,), dtype=np.float32)
        PCA_8_ds = h.create_or_recreate_dataset(outfile, "PCA_8", (Num_of_proteins,), dtype=np.float32)
        PCA_9_ds = h.create_or_recreate_dataset(outfile, "PCA_9", (Num_of_proteins,), dtype=np.float32)
        PCA_10_ds = h.create_or_recreate_dataset(outfile, "PCA_10", (Num_of_proteins,), dtype=np.float32)
        PCA_11_ds = h.create_or_recreate_dataset(outfile, "PCA_11", (Num_of_proteins,), dtype=np.float32)
        PCA_12_ds = h.create_or_recreate_dataset(outfile, "PCA_12", (Num_of_proteins,), dtype=np.float32)
        PCA_13_ds = h.create_or_recreate_dataset(outfile, "PCA_13", (Num_of_proteins,), dtype=np.float32)
        PCA_14_ds = h.create_or_recreate_dataset(outfile, "PCA_14", (Num_of_proteins,), dtype=np.float32)
        PCA_15_ds = h.create_or_recreate_dataset(outfile, "PCA_15", (Num_of_proteins,), dtype=np.float32)
        PCA_16_ds = h.create_or_recreate_dataset(outfile, "PCA_16", (Num_of_proteins,), dtype=np.float32)
        PCA_17_ds = h.create_or_recreate_dataset(outfile, "PCA_17", (Num_of_proteins,), dtype=np.float32)
        PCA_18_ds = h.create_or_recreate_dataset(outfile, "PCA_18", (Num_of_proteins,), dtype=np.float32)
        PCA_19_ds = h.create_or_recreate_dataset(outfile, "PCA_19", (Num_of_proteins,), dtype=np.float32)
        PCA_20_ds = h.create_or_recreate_dataset(outfile, "PCA_20", (Num_of_proteins,), dtype=np.float32)
        PCA_21_ds = h.create_or_recreate_dataset(outfile, "PCA_21", (Num_of_proteins,), dtype=np.float32)
        PCA_22_ds = h.create_or_recreate_dataset(outfile, "PCA_22", (Num_of_proteins,), dtype=np.float32)
        PCA_23_ds = h.create_or_recreate_dataset(outfile, "PCA_23", (Num_of_proteins,), dtype=np.float32)
        PCA_24_ds = h.create_or_recreate_dataset(outfile, "PCA_24", (Num_of_proteins,), dtype=np.float32)
        PCA_25_ds = h.create_or_recreate_dataset(outfile, "PCA_25", (Num_of_proteins,), dtype=np.float32)
        PCA_26_ds = h.create_or_recreate_dataset(outfile, "PCA_26", (Num_of_proteins,), dtype=np.float32)
        PCA_27_ds = h.create_or_recreate_dataset(outfile, "PCA_27", (Num_of_proteins,), dtype=np.float32)
        PCA_28_ds = h.create_or_recreate_dataset(outfile, "PCA_28", (Num_of_proteins,), dtype=np.float32)
        PCA_29_ds = h.create_or_recreate_dataset(outfile, "PCA_29", (Num_of_proteins,), dtype=np.float32)
        PCA_30_ds = h.create_or_recreate_dataset(outfile, "PCA_30", (Num_of_proteins,), dtype=np.float32)
        PCA_31_ds = h.create_or_recreate_dataset(outfile, "PCA_31", (Num_of_proteins,), dtype=np.float32)
        PCA_32_ds = h.create_or_recreate_dataset(outfile, "PCA_32", (Num_of_proteins,), dtype=np.float32)
        PCA_33_ds = h.create_or_recreate_dataset(outfile, "PCA_33", (Num_of_proteins,), dtype=np.float32)
        PCA_34_ds = h.create_or_recreate_dataset(outfile, "PCA_34", (Num_of_proteins,), dtype=np.float32)
        PCA_35_ds = h.create_or_recreate_dataset(outfile, "PCA_35", (Num_of_proteins,), dtype=np.float32)
        PCA_36_ds = h.create_or_recreate_dataset(outfile, "PCA_36", (Num_of_proteins,), dtype=np.float32)


        ###### protein data ######
        pdb_code_ds = h.create_or_recreate_dataset(outfile,'PDB_code', (Num_of_proteins,), dtype=string_type)
        release_year_ds = h.create_or_recreate_dataset(outfile,'release_year', (Num_of_proteins,), dtype=string_type)
        ec_number_ds = h.create_or_recreate_dataset(outfile,'EC_number', (Num_of_proteins,), dtype=string_type)
        protein_name_ds = h.create_or_recreate_dataset(outfile,'protein_name', (Num_of_proteins,), dtype=string_type)
        logkd_ki_ds = h.create_or_recreate_dataset(outfile,'-logKd_over_Ki', (Num_of_proteins,), dtype=np.float32)
        kd_ki_ds = h.create_or_recreate_dataset(outfile,'Kd_over_Ki', (Num_of_proteins,), dtype=string_type)
        reference_ds = h.create_or_recreate_dataset(outfile,'reference', (Num_of_proteins,), dtype=string_type)
        if do_cluster:
            cluster_id_ds = outfile.create_dataset('cluster ID', (Num_of_proteins,), dtype=string_type)
        resolution_ds = outfile.create_dataset('resolution', (Num_of_proteins,), dtype=np.float32)
        ligand_ds = outfile.create_dataset('ligand_name', (Num_of_proteins,), dtype=string_type)

        for mol_idx in range(Num_of_proteins):
            if mol_idx % 50 == 0:
                print('Got to Molecule no. ', mol_idx, PDB_List[mol_idx])
            molID_ds[mol_idx] = mol_idx

            # get the current PDB code
            current_code=PDB_List[mol_idx]
            # get the rows in the dataframes
            current_index_row=df_index.loc[df_index['PDB_code']==current_code]
            print(current_code)
            print(current_index_row)
            current_data_row=df_data.loc[df_data['PDB_code']==current_code]
            print(current_data_row)
            if do_cluster:
                current_cluster_row=df_cluster.loc[df_cluster['PDB_code']==current_code]
                print(current_cluster_row)
            pdb_code = current_index_row.iloc[0]['PDB_code']
            pdb_code_ds[mol_idx] = np.array(pdb_code,dtype=string_type)

            #      Persistence entropy
            (P_pers_S_1_ds[mol_idx], P_pers_S_2_ds[mol_idx], P_pers_S_3_ds[mol_idx],
            P_no_p_1_ds[mol_idx], P_no_p_2_ds[mol_idx],P_no_p_3_ds[mol_idx],
            P_bottle_1_ds[mol_idx],P_bottle_2_ds[mol_idx],P_bottle_3_ds[mol_idx],
            P_wasser_1_ds[mol_idx],P_wasser_2_ds[mol_idx], P_wasser_3_ds[mol_idx],
            P_landsc_1_ds[mol_idx], P_landsc_2_ds[mol_idx],P_landsc_3_ds[mol_idx],
            P_pers_img_1_ds[mol_idx],P_pers_img_2_ds[mol_idx],P_pers_img_3_ds[mol_idx],
            L_pers_S_1_ds[mol_idx],L_pers_S_2_ds[mol_idx],L_pers_S_3_ds[mol_idx],
            L_no_p_1_ds[mol_idx],L_no_p_2_ds[mol_idx],L_no_p_3_ds[mol_idx],
            L_bottle_1_ds[mol_idx],L_bottle_2_ds[mol_idx],L_bottle_3_ds[mol_idx],
            L_wasser_1_ds[mol_idx],L_wasser_2_ds[mol_idx],L_wasser_3_ds[mol_idx],
            L_landsc_1_ds[mol_idx],L_landsc_2_ds[mol_idx],L_landsc_3_ds[mol_idx],
            L_pers_img_1_ds[mol_idx],L_pers_img_2_ds[mol_idx],L_pers_img_3_ds[mol_idx]) = topl_PDB_large[mol_idx]

            #PCA
            (PCA_1_ds[mol_idx], PCA_2_ds[mol_idx], PCA_3_ds[mol_idx], PCA_4_ds[mol_idx], PCA_5_ds[mol_idx],
            PCA_6_ds[mol_idx], PCA_7_ds[mol_idx], PCA_8_ds[mol_idx], PCA_9_ds[mol_idx], PCA_10_ds[mol_idx],
            PCA_11_ds[mol_idx], PCA_12_ds[mol_idx], PCA_13_ds[mol_idx], PCA_14_ds[mol_idx], PCA_15_ds[mol_idx],
            PCA_16_ds[mol_idx], PCA_17_ds[mol_idx], PCA_18_ds[mol_idx], PCA_19_ds[mol_idx], PCA_20_ds[mol_idx],
            PCA_21_ds[mol_idx], PCA_22_ds[mol_idx], PCA_23_ds[mol_idx], PCA_24_ds[mol_idx], PCA_25_ds[mol_idx],
            PCA_26_ds[mol_idx], PCA_27_ds[mol_idx], PCA_28_ds[mol_idx], PCA_29_ds[mol_idx], PCA_30_ds[mol_idx],
            PCA_31_ds[mol_idx], PCA_32_ds[mol_idx], PCA_33_ds[mol_idx], PCA_34_ds[mol_idx], PCA_35_ds[mol_idx],
            PCA_36_ds[mol_idx])=principalComponents_core[mol_idx]

            release_year = current_index_row.iloc[0]['release_year']
                    #print(type(release_year))
            release_year_ds[mol_idx] = release_year
            ec_number = current_index_row.iloc[0]['EC_number']
            ec_number_ds[mol_idx] = ec_number
            protein_name = current_index_row.iloc[0]['protein_name']
            protein_name_ds[mol_idx] = protein_name
            logkd_ki = current_data_row.iloc[0]['-logKd/Ki']
            logkd_ki_ds[mol_idx] = float(logkd_ki)
            kd_ki = current_data_row.iloc[0]['Kd/Ki']
            kd_ki_ds[mol_idx] = kd_ki
            reference = current_data_row.iloc[0]['reference']
            reference_ds[mol_idx] = reference
            if do_cluster:
                cluster_id = current_cluster_row.iloc[0]['cluster ID']
                cluster_id_ds[mol_idx] = cluster_id
            resolution = current_data_row.iloc[0]['resolution']
            resolution_ds[mol_idx] = float(resolution)
            ligand_name = current_data_row.iloc[0]['ligand name']
            ligand_ds[mol_idx] = ligand_name
        outfile.close()

make_dataset = False
topl_PDB_all_refined_mat_large = np.genfromtxt('output.csv', delimiter=',')
topl_PDB_all_refined_large=topl_PDB_all_refined_mat_large.tolist()
from sklearn.decomposition import PCA
pca = PCA(n_components=36)
principalComponents_refined = pca.fit_transform(topl_PDB_all_refined_large)

if make_dataset:
    outfile = h5py.File(os.path.join(save_dir, out_file_name), "w")
    string_type = h5py.string_dtype(encoding='utf-8')

    PDB_List = df_index_refined['PDB_code']
    df_index = df_index_refined
    df_data = df_data_refined
    # df_cluster = df_cluster_core
    topl_PDB_large = topl_PDB_all_refined_large

    num_of_proteins_override = 0  # 21
    do_cluster = False

    ## NTS! Put the PCA calculation in here !

    if True:

        if num_of_proteins_override == 0:
            # do all proteins woo
            Num_of_proteins = len(PDB_List)
        else:
            Num_of_proteins = num_of_proteins_override

        ##################### set up the output datasets ################################

        ## this sets up the output datasets
        molID_ds = h.create_or_recreate_dataset(outfile, "molID", (Num_of_proteins,), dtype=np.int8)
        ###### topological data ####
        ###### proteins #####
        #      Persistence entropy
        P_pers_S_1_ds = h.create_or_recreate_dataset(outfile, "P_pers_S_1", (Num_of_proteins,), dtype=np.float32)
        P_pers_S_2_ds = h.create_or_recreate_dataset(outfile, "P_pers_S_2", (Num_of_proteins,), dtype=np.float32)
        P_pers_S_3_ds = h.create_or_recreate_dataset(outfile, "P_pers_S_3", (Num_of_proteins,), dtype=np.float32)
        #      No. of points
        P_no_p_1_ds = h.create_or_recreate_dataset(outfile, "P_no_p_1", (Num_of_proteins,), dtype=np.float32)
        P_no_p_2_ds = h.create_or_recreate_dataset(outfile, "P_no_p_2", (Num_of_proteins,), dtype=np.float32)
        P_no_p_3_ds = h.create_or_recreate_dataset(outfile, "P_no_p_3", (Num_of_proteins,), dtype=np.float32)
        #      Bottleneck
        P_bottle_1_ds = h.create_or_recreate_dataset(outfile, "P_bottle_1", (Num_of_proteins,), dtype=np.float32)
        P_bottle_2_ds = h.create_or_recreate_dataset(outfile, "P_bottle_2", (Num_of_proteins,), dtype=np.float32)
        P_bottle_3_ds = h.create_or_recreate_dataset(outfile, "P_bottle_3", (Num_of_proteins,), dtype=np.float32)
        #      Wasserstein
        P_wasser_1_ds = h.create_or_recreate_dataset(outfile, "P_wasser_1", (Num_of_proteins,), dtype=np.float32)
        P_wasser_2_ds = h.create_or_recreate_dataset(outfile, "P_wasser_2", (Num_of_proteins,), dtype=np.float32)
        P_wasser_3_ds = h.create_or_recreate_dataset(outfile, "P_wasser_3", (Num_of_proteins,), dtype=np.float32)
        #      landscape
        P_landsc_1_ds = h.create_or_recreate_dataset(outfile, "P_landsc_1", (Num_of_proteins,), dtype=np.float32)
        P_landsc_2_ds = h.create_or_recreate_dataset(outfile, "P_landsc_2", (Num_of_proteins,), dtype=np.float32)
        P_landsc_3_ds = h.create_or_recreate_dataset(outfile, "P_landsc_3", (Num_of_proteins,), dtype=np.float32)
        #      persistence image
        P_pers_img_1_ds = h.create_or_recreate_dataset(outfile, "P_pers_img_1", (Num_of_proteins,), dtype=np.float32)
        P_pers_img_2_ds = h.create_or_recreate_dataset(outfile, "P_pers_img_2", (Num_of_proteins,), dtype=np.float32)
        P_pers_img_3_ds = h.create_or_recreate_dataset(outfile, "P_pers_img_3", (Num_of_proteins,), dtype=np.float32)
        #### ligands ####
        #      Persistence entropy
        L_pers_S_1_ds = h.create_or_recreate_dataset(outfile, "L_pers_S_1", (Num_of_proteins,), dtype=np.float32)
        L_pers_S_2_ds = h.create_or_recreate_dataset(outfile, "L_pers_S_2", (Num_of_proteins,), dtype=np.float32)
        L_pers_S_3_ds = h.create_or_recreate_dataset(outfile, "L_pers_S_3", (Num_of_proteins,), dtype=np.float32)
        #      No. of points
        L_no_p_1_ds = h.create_or_recreate_dataset(outfile, "L_no_p_1", (Num_of_proteins,), dtype=np.float32)
        L_no_p_2_ds = h.create_or_recreate_dataset(outfile, "L_no_p_2", (Num_of_proteins,), dtype=np.float32)
        L_no_p_3_ds = h.create_or_recreate_dataset(outfile, "L_no_p_3", (Num_of_proteins,), dtype=np.float32)
        #      Bottleneck
        L_bottle_1_ds = h.create_or_recreate_dataset(outfile, "L_bottle_1", (Num_of_proteins,), dtype=np.float32)
        L_bottle_2_ds = h.create_or_recreate_dataset(outfile, "L_bottle_2", (Num_of_proteins,), dtype=np.float32)
        L_bottle_3_ds = h.create_or_recreate_dataset(outfile, "L_bottle_3", (Num_of_proteins,), dtype=np.float32)
        #      Wasserstein
        L_wasser_1_ds = h.create_or_recreate_dataset(outfile, "L_wasser_1", (Num_of_proteins,), dtype=np.float32)
        L_wasser_2_ds = h.create_or_recreate_dataset(outfile, "L_wasser_2", (Num_of_proteins,), dtype=np.float32)
        L_wasser_3_ds = h.create_or_recreate_dataset(outfile, "L_wasser_3", (Num_of_proteins,), dtype=np.float32)
        #      landscape
        L_landsc_1_ds = h.create_or_recreate_dataset(outfile, "L_landsc_1", (Num_of_proteins,), dtype=np.float32)
        L_landsc_2_ds = h.create_or_recreate_dataset(outfile, "L_landsc_2", (Num_of_proteins,), dtype=np.float32)
        L_landsc_3_ds = h.create_or_recreate_dataset(outfile, "L_landsc_3", (Num_of_proteins,), dtype=np.float32)
        #      persistence image
        L_pers_img_1_ds = h.create_or_recreate_dataset(outfile, "L_pers_img_1", (Num_of_proteins,), dtype=np.float32)
        L_pers_img_2_ds = h.create_or_recreate_dataset(outfile, "L_pers_img_2", (Num_of_proteins,), dtype=np.float32)
        L_pers_img_3_ds = h.create_or_recreate_dataset(outfile, "L_pers_img_3", (Num_of_proteins,), dtype=np.float32)

        PCA_1_ds = h.create_or_recreate_dataset(outfile, "PCA_1", (Num_of_proteins,), dtype=np.float32)
        PCA_2_ds = h.create_or_recreate_dataset(outfile, "PCA_2", (Num_of_proteins,), dtype=np.float32)
        PCA_3_ds = h.create_or_recreate_dataset(outfile, "PCA_3", (Num_of_proteins,), dtype=np.float32)
        PCA_4_ds = h.create_or_recreate_dataset(outfile, "PCA_4", (Num_of_proteins,), dtype=np.float32)
        PCA_5_ds = h.create_or_recreate_dataset(outfile, "PCA_5", (Num_of_proteins,), dtype=np.float32)
        PCA_6_ds = h.create_or_recreate_dataset(outfile, "PCA_6", (Num_of_proteins,), dtype=np.float32)
        PCA_7_ds = h.create_or_recreate_dataset(outfile, "PCA_7", (Num_of_proteins,), dtype=np.float32)
        PCA_8_ds = h.create_or_recreate_dataset(outfile, "PCA_8", (Num_of_proteins,), dtype=np.float32)
        PCA_9_ds = h.create_or_recreate_dataset(outfile, "PCA_9", (Num_of_proteins,), dtype=np.float32)
        PCA_10_ds = h.create_or_recreate_dataset(outfile, "PCA_10", (Num_of_proteins,), dtype=np.float32)
        PCA_11_ds = h.create_or_recreate_dataset(outfile, "PCA_11", (Num_of_proteins,), dtype=np.float32)
        PCA_12_ds = h.create_or_recreate_dataset(outfile, "PCA_12", (Num_of_proteins,), dtype=np.float32)
        PCA_13_ds = h.create_or_recreate_dataset(outfile, "PCA_13", (Num_of_proteins,), dtype=np.float32)
        PCA_14_ds = h.create_or_recreate_dataset(outfile, "PCA_14", (Num_of_proteins,), dtype=np.float32)
        PCA_15_ds = h.create_or_recreate_dataset(outfile, "PCA_15", (Num_of_proteins,), dtype=np.float32)
        PCA_16_ds = h.create_or_recreate_dataset(outfile, "PCA_16", (Num_of_proteins,), dtype=np.float32)
        PCA_17_ds = h.create_or_recreate_dataset(outfile, "PCA_17", (Num_of_proteins,), dtype=np.float32)
        PCA_18_ds = h.create_or_recreate_dataset(outfile, "PCA_18", (Num_of_proteins,), dtype=np.float32)
        PCA_19_ds = h.create_or_recreate_dataset(outfile, "PCA_19", (Num_of_proteins,), dtype=np.float32)
        PCA_20_ds = h.create_or_recreate_dataset(outfile, "PCA_20", (Num_of_proteins,), dtype=np.float32)
        PCA_21_ds = h.create_or_recreate_dataset(outfile, "PCA_21", (Num_of_proteins,), dtype=np.float32)
        PCA_22_ds = h.create_or_recreate_dataset(outfile, "PCA_22", (Num_of_proteins,), dtype=np.float32)
        PCA_23_ds = h.create_or_recreate_dataset(outfile, "PCA_23", (Num_of_proteins,), dtype=np.float32)
        PCA_24_ds = h.create_or_recreate_dataset(outfile, "PCA_24", (Num_of_proteins,), dtype=np.float32)
        PCA_25_ds = h.create_or_recreate_dataset(outfile, "PCA_25", (Num_of_proteins,), dtype=np.float32)
        PCA_26_ds = h.create_or_recreate_dataset(outfile, "PCA_26", (Num_of_proteins,), dtype=np.float32)
        PCA_27_ds = h.create_or_recreate_dataset(outfile, "PCA_27", (Num_of_proteins,), dtype=np.float32)
        PCA_28_ds = h.create_or_recreate_dataset(outfile, "PCA_28", (Num_of_proteins,), dtype=np.float32)
        PCA_29_ds = h.create_or_recreate_dataset(outfile, "PCA_29", (Num_of_proteins,), dtype=np.float32)
        PCA_30_ds = h.create_or_recreate_dataset(outfile, "PCA_30", (Num_of_proteins,), dtype=np.float32)
        PCA_31_ds = h.create_or_recreate_dataset(outfile, "PCA_31", (Num_of_proteins,), dtype=np.float32)
        PCA_32_ds = h.create_or_recreate_dataset(outfile, "PCA_32", (Num_of_proteins,), dtype=np.float32)
        PCA_33_ds = h.create_or_recreate_dataset(outfile, "PCA_33", (Num_of_proteins,), dtype=np.float32)
        PCA_34_ds = h.create_or_recreate_dataset(outfile, "PCA_34", (Num_of_proteins,), dtype=np.float32)
        PCA_35_ds = h.create_or_recreate_dataset(outfile, "PCA_35", (Num_of_proteins,), dtype=np.float32)
        PCA_36_ds = h.create_or_recreate_dataset(outfile, "PCA_36", (Num_of_proteins,), dtype=np.float32)

        ###### protein data ######
        pdb_code_ds = h.create_or_recreate_dataset(outfile, 'PDB_code', (Num_of_proteins,), dtype=string_type)
        release_year_ds = h.create_or_recreate_dataset(outfile, 'release_year', (Num_of_proteins,), dtype=string_type)
        ec_number_ds = h.create_or_recreate_dataset(outfile, 'EC_number', (Num_of_proteins,), dtype=string_type)
        protein_name_ds = h.create_or_recreate_dataset(outfile, 'protein_name', (Num_of_proteins,), dtype=string_type)
        logkd_ki_ds = h.create_or_recreate_dataset(outfile, '-logKd_over_Ki', (Num_of_proteins,), dtype=np.float32)
        kd_ki_ds = h.create_or_recreate_dataset(outfile, 'Kd_over_Ki', (Num_of_proteins,), dtype=string_type)
        reference_ds = h.create_or_recreate_dataset(outfile, 'reference', (Num_of_proteins,), dtype=string_type)
        if do_cluster:
            cluster_id_ds = outfile.create_dataset('cluster ID', (Num_of_proteins,), dtype=string_type)
        resolution_ds = outfile.create_dataset('resolution', (Num_of_proteins,), dtype=np.float32)
        ligand_ds = outfile.create_dataset('ligand_name', (Num_of_proteins,), dtype=string_type)

        for mol_idx in range(Num_of_proteins):
            if mol_idx % 50 == 0:
                print('Got to Molecule no. ', mol_idx, PDB_List[mol_idx])
            molID_ds[mol_idx] = mol_idx

            # get the current PDB code
            current_code = PDB_List[mol_idx]
            # get the rows in the dataframes
            current_index_row = df_index.loc[df_index['PDB_code'] == current_code]
            print(current_code)
            print(current_index_row)
            current_data_row = df_data.loc[df_data['PDB_code'] == current_code]
            print(current_data_row)
            if do_cluster:
                current_cluster_row = df_cluster.loc[df_cluster['PDB_code'] == current_code]
                print(current_cluster_row)
            pdb_code = current_index_row.iloc[0]['PDB_code']
            pdb_code_ds[mol_idx] = np.array(pdb_code, dtype=string_type)

            #      Persistence entropy
            (P_pers_S_1_ds[mol_idx], P_pers_S_2_ds[mol_idx], P_pers_S_3_ds[mol_idx],
             P_no_p_1_ds[mol_idx], P_no_p_2_ds[mol_idx], P_no_p_3_ds[mol_idx],
             P_bottle_1_ds[mol_idx], P_bottle_2_ds[mol_idx], P_bottle_3_ds[mol_idx],
             P_wasser_1_ds[mol_idx], P_wasser_2_ds[mol_idx], P_wasser_3_ds[mol_idx],
             P_landsc_1_ds[mol_idx], P_landsc_2_ds[mol_idx], P_landsc_3_ds[mol_idx],
             P_pers_img_1_ds[mol_idx], P_pers_img_2_ds[mol_idx], P_pers_img_3_ds[mol_idx],
             L_pers_S_1_ds[mol_idx], L_pers_S_2_ds[mol_idx], L_pers_S_3_ds[mol_idx],
             L_no_p_1_ds[mol_idx], L_no_p_2_ds[mol_idx], L_no_p_3_ds[mol_idx],
             L_bottle_1_ds[mol_idx], L_bottle_2_ds[mol_idx], L_bottle_3_ds[mol_idx],
             L_wasser_1_ds[mol_idx], L_wasser_2_ds[mol_idx], L_wasser_3_ds[mol_idx],
             L_landsc_1_ds[mol_idx], L_landsc_2_ds[mol_idx], L_landsc_3_ds[mol_idx],
             L_pers_img_1_ds[mol_idx], L_pers_img_2_ds[mol_idx], L_pers_img_3_ds[mol_idx]) = topl_PDB_large[mol_idx]

            # PCA
            (PCA_1_ds[mol_idx], PCA_2_ds[mol_idx], PCA_3_ds[mol_idx], PCA_4_ds[mol_idx], PCA_5_ds[mol_idx],
             PCA_6_ds[mol_idx], PCA_7_ds[mol_idx], PCA_8_ds[mol_idx], PCA_9_ds[mol_idx], PCA_10_ds[mol_idx],
             PCA_11_ds[mol_idx], PCA_12_ds[mol_idx], PCA_13_ds[mol_idx], PCA_14_ds[mol_idx], PCA_15_ds[mol_idx],
             PCA_16_ds[mol_idx], PCA_17_ds[mol_idx], PCA_18_ds[mol_idx], PCA_19_ds[mol_idx], PCA_20_ds[mol_idx],
             PCA_21_ds[mol_idx], PCA_22_ds[mol_idx], PCA_23_ds[mol_idx], PCA_24_ds[mol_idx], PCA_25_ds[mol_idx],
             PCA_26_ds[mol_idx], PCA_27_ds[mol_idx], PCA_28_ds[mol_idx], PCA_29_ds[mol_idx], PCA_30_ds[mol_idx],
             PCA_31_ds[mol_idx], PCA_32_ds[mol_idx], PCA_33_ds[mol_idx], PCA_34_ds[mol_idx], PCA_35_ds[mol_idx],
             PCA_36_ds[mol_idx]) = principalComponents_refined[mol_idx]

            release_year = current_index_row.iloc[0]['release_year']
            # print(type(release_year))
            release_year_ds[mol_idx] = release_year
            ec_number = current_index_row.iloc[0]['EC_number']
            ec_number_ds[mol_idx] = ec_number
            protein_name = current_index_row.iloc[0]['protein_name']
            protein_name_ds[mol_idx] = protein_name
            logkd_ki = current_data_row.iloc[0]['-logKd/Ki']
            logkd_ki_ds[mol_idx] = float(logkd_ki)
            kd_ki = current_data_row.iloc[0]['Kd/Ki']
            kd_ki_ds[mol_idx] = kd_ki
            reference = current_data_row.iloc[0]['reference']
            reference_ds[mol_idx] = reference
            if do_cluster:
                cluster_id = current_cluster_row.iloc[0]['cluster ID']
                cluster_id_ds[mol_idx] = cluster_id
            resolution = current_data_row.iloc[0]['resolution']
            resolution_ds[mol_idx] = float(resolution)
            ligand_name = current_data_row.iloc[0]['ligand name']
            ligand_ds[mol_idx] = ligand_name
        outfile.close()



