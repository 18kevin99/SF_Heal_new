from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
# import gym_molecule
import copy
import networkx as nx
from gym_molecule.envs.sascorer import calculateScore
from gym_molecule.dataset.dataset_utils import gdb_dataset,mol_to_nx,nx_to_mol
import random
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from contextlib import contextmanager
import sys, os
import tensorflow.compat.v1 as tf
import gym_molecule.models.reinvent.model as mm

model = mm.Model.load_from_file('/data/data/kcoutinh/Research/rl_graph_generation/gym-molecule/gym_molecule/kev_models/model_zinc_250.78')

def read_smi_file(file_path):
    """
    Reads a SMILES file.
    :param file_path: Path to a SMILES file.
    :return: A list with all the SMILES.
    """
    with open(file_path, "r") as smi_file:
        return [smi.rstrip().split(",")[0] for smi in smi_file]
    

dataset = read_smi_file('/data/data/kcoutinh/Research/rl_graph_generation/gym-molecule/gym_molecule/dataset/sampled_smiles_2.smi')

df_org = pd.DataFrame(dataset, columns=['SMILES'])

log_lh = []

for mols in df_org['SMILES']:
    try:
        log_lh.append(model.likelihood(mols))
    except:
        log_lh.append(1000)

df_org["Log_likeli"] = log_lh

sns.histplot(data = df_org, x = log_lh, bins = 30)
plt.show()