#################################################################################################
# This code was copied from (https://github.com/DeepSynthesis/ScopeMap/tree/main/SubstrateScore)
#################################################################################################

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)


def spoc_descriptors(file_path):
    # Extract name from file path (remove extension)
    name = os.path.splitext(os.path.basename(file_path))[0]
    
    # data loading
    data = pd.read_csv(file_path, encoding='utf-8')
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    data['Mol_r_al'] = data['smiles'].apply(Chem.MolFromSmiles)


    # Morgan
    X_r_al = []
    X_r_am = []
    X_p = []
    MACC_al = []
    MACC_am = []
    MACC_p = []
    rdkit_al = []
    rdkit_am = []
    rdkit_p = []
    for i in range(len(data)):
        try:
            r_al = AllChem.GetMorganFingerprintAsBitVect(
            data['Mol_r_al'][i], 4, nBits=1024)
            r_al_MACC = MACCSkeys.GenMACCSKeys(data['Mol_r_al'][i])
            r_al_rdkit = calculator.CalcDescriptors(data['Mol_r_al'][i])
        except:
            print('ERROR Aldehyde Reactant: '+data['smiles'][i])
            continue
        

        X_r_al.append(r_al)
        MACC_al.append(r_al_MACC)
        rdkit_al.append(r_al_rdkit)


    X_r_al = np.array(X_r_al)
    MACC_al = np.array(MACC_al)
    rdkit_al = np.array(rdkit_al)

    # X_r = np.array([AllChem.GetMorganFingerprintAsBitVect(
    #     mol, 4, nBits=1024) for mol in data['Mol_r']])
    # X_p = np.array([AllChem.GetMorganFingerprintAsBitVect(
    #     mol, 4, nBits=1024) for mol in data['Mol_p']])


    # data['fp'] = data["Mol"].apply(morgan)
    # X = data['fp'].values

    # X = MACC.copy()

    X_r_al = np.hstack((MACC_al, X_r_al))

    # X_r_al = MACC_al

    # X_r_al = X_r_al

    # X_r_al = np.hstack((MACC_al, rdkit_al))
    # X_r_am = np.hstack((MACC_am, rdkit_am))
    # X_p = np.hstack((MACC_p, rdkit_p))

    # X_r_al = np.hstack((X_r_al, rdkit_al))
    # X_r_am = np.hstack((X_r_am, rdkit_am))
    # X_p = np.hstack((X_p, rdkit_p))

    # X_r_p = X_r_al - X_p

    # transfer NaN and inf to zero
    where_are_nan = np.isnan(X_r_al)
    where_are_inf = np.isinf(X_r_al)
    X_r_al[where_are_nan] = 0
    X_r_al[where_are_inf] = 0

    df_al = pd.DataFrame(X_r_al, columns=[str(i) for i in range(len(X_r_al[0]))])

    df_al.to_csv('fp_spoc_morgan41024_Maccs_'+name+'.csv', index=False)




if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python desc_gen_util.py <file_path>")
        print("Example: python desc_gen_util.py data.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    if not file_path.endswith('.csv'):
        print("Error: Input file must be a CSV file.")
        sys.exit(1)
    
    print(f"Processing file: {file_path}")
    spoc_descriptors(file_path)
    print("Processing completed.")
