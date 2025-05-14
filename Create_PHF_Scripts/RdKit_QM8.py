######################################################################################################################
##########################   make RDKIT features and put them into a csvfile #################################
######################################################################################################################
##########################   QM8    QM8     QM8 QM8       #################################
######################################################################################################################

# makes rdkit features for datasets

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
import pandas as pd
import os
import sys
from src import helper_functions as h

dataset_name = 'qm8'
print(f'Using dataset {dataset_name}')

Failures = []  #

save_dir = r'C:\Users\eg16993\OneDrive - University of Bristol\Documents\Datasets\topol_datasets'
data_dir = r'C:\Users\eg16993\OneDrive - University of Bristol\Documents\Datasets'
results_dir = r"C:\Users\eg16993\OneDrive - University of Bristol\Documents\Results\graphs_and_topology\d_" + dataset_name

dataset_file = dataset_name + '.csv'
output_file = dataset_name + '_rdkit.csv'

og_dataset = pd.read_csv(os.path.join(data_dir, dataset_file))


features, failures = h.make_rdkit_dataset(dataset_name,
                       dataset_file=dataset_file,
                       output_file=output_file,
                       data_dir=data_dir,
                       save_dir=save_dir,
                       column_heading_name=[],
                       column_heading_smiles=[])
