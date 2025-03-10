
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from biopandas.pdb import PandasPdb
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
#logging.basicConfig(encoding='utf-8', level=logging.INFO)

class TOUGH_M1_dataset(Dataset):
    def __init__(self,pdb_list,data_root="./TOUGH-M1/"):
        self.data_root = data_root
        self.dataset_path = data_root + "TOUGH-M1_dataset/"
        self.pdb_list = pdb_list
        self.data = self.preprocess()

        # map pdb code -> entry
        self._pdb_map = {}
        for i, pdb_entry in enumerate(self.pdb_list):
            code = pdb_entry['code5'] if 'code5' in pdb_entry else pdb_entry['code']
            self._pdb_map[code] = i
        
    def __len__(self):
        return len(self.data)

    def preprocess(self):
        logger.debug("TOUGH_M1_dataset preprocessing")
        data = []
        for idx in tqdm(range(len(self.pdb_list))):
            id = self.pdb_list[idx]["code5"]
            data.append(self.get_data_per_protein(id))
        logger.info("TOUGH_M1_dataset preprocess done")
        return data
        
    def get_data_per_protein(self,id):
        logger.debug(f"Get data of protein id:{id}")
        dir = self.dataset_path+id
        receptor= PandasPdb().read_pdb(dir+f'/{id}.pdb')
        ligand = PandasPdb().read_pdb(dir+f'/{id}00.pdb')
        f_text = subprocess.run("obabel -ipdb "+dir+f'/{id}.pdb'+" -omol2", shell=True, stdout=subprocess.PIPE , stderr=subprocess.PIPE ,encoding="utf-8").stdout
        bond_start = f_text.find('@<TRIPOS>BOND')
        bond_end = -1
        df_bonds = f_text[bond_start:bond_end].replace('@<TRIPOS>BOND\n', '')
        df_bonds = np.array([x for x in df_bonds.split()]).reshape(-1,4)
        df_bonds = pd.DataFrame(df_bonds, columns=['bond_id', 'atom1', 'atom2', 'bond_type'])
        def str_to_int(x):
            return int(x)
        df_bonds['bond_id'] = df_bonds['bond_id'].apply(str_to_int)
        df_bonds['atom1'] = df_bonds['atom1'].apply(str_to_int)
        df_bonds['atom2'] = df_bonds['atom2'].apply(str_to_int)

        # 座標
        receptor_coords = receptor.df['ATOM'][['x_coord','y_coord','z_coord']].to_numpy()
        ligand_coords = ligand.df['HETATM'][['x_coord','y_coord','z_coord']].to_numpy()
    
        # 距離行列からポケット部位抽出
        dist_mat = distance_matrix(ligand_coords,receptor_coords)
        dist_atom_to_ligand = np.min(dist_mat,axis=0)
        pocket_residue_numbers = list(receptor.df['ATOM'][dist_atom_to_ligand < 8]['residue_number'].unique())

        def pocket_flag(x):
            return 1 if x in pocket_residue_numbers else 0
        receptor.df['ATOM']['pocket_flag'] = receptor.df['ATOM']['residue_number'].apply(pocket_flag)
    
        # 結合情報
        def create_init_bond_dict():
            return {'atom':[],'bond_type':[]}
        
        bond_dict = defaultdict()
        for i in range(len(df_bonds)):
            atom1 = df_bonds['atom1'][i]
            atom2 = df_bonds['atom2'][i]
            bond_type = df_bonds['bond_type'][i]
            bond_dict[atom1-1] = create_init_bond_dict()
            bond_dict[atom2-1] = create_init_bond_dict()
            bond_dict[atom1-1]['atom'].append(atom2-1)
            bond_dict[atom1-1]['bond_type'].append(bond_type)
            bond_dict[atom2-1]['atom'].append(atom1-1)
            bond_dict[atom2-1]['bond_type'].append(bond_type)
    
        # 特徴量抽出
        receptor_pocket_flag = receptor.df['ATOM']['pocket_flag'].to_numpy().astype(np.float64)
        receptor_atom_feature = np.array([
            receptor.df['ATOM']['element_symbol']=='N',
            receptor.df['ATOM']['element_symbol']=='C',
            receptor.df['ATOM']['element_symbol']=='O',
            receptor.df['ATOM']['element_symbol']=='S',
            receptor.df['ATOM']['residue_name']=='LYS', 
            receptor.df['ATOM']['residue_name']=='ASP', 
            receptor.df['ATOM']['residue_name']=='SER', 
            receptor.df['ATOM']['residue_name']=='PRO', 
            receptor.df['ATOM']['residue_name']=='PHE', 
            receptor.df['ATOM']['residue_name']=='TYR', 
            receptor.df['ATOM']['residue_name']=='GLN', 
            receptor.df['ATOM']['residue_name']=='HIS', 
            receptor.df['ATOM']['residue_name']=='LEU',
            receptor.df['ATOM']['residue_name']=='GLY', 
            receptor.df['ATOM']['residue_name']=='GLU', 
            receptor.df['ATOM']['residue_name']=='ILE', 
            receptor.df['ATOM']['residue_name']=='CYS', 
            receptor.df['ATOM']['residue_name']=='ARG', 
            receptor.df['ATOM']['residue_name']=='VAL', 
            receptor.df['ATOM']['residue_name']=='ASN', 
            receptor.df['ATOM']['residue_name']=='ALA', 
            receptor.df['ATOM']['residue_name']=='MET',
            receptor.df['ATOM']['residue_name']=='THR', 
            receptor.df['ATOM']['residue_name']=='TRP'
        ]).astype(np.float64).T
        receptor_residue_number = receptor.df['ATOM']['residue_number'].to_numpy()
    
        # torch.tensor化
        receptor_coords = torch.from_numpy(receptor_coords).float()
        receptor_pocket_flag = torch.from_numpy(receptor_pocket_flag).float()
        receptor_atom_feature = torch.from_numpy(receptor_atom_feature).float()
        receptor_residue_number = torch.from_numpy(receptor_residue_number)

        return {
            'pocket_id':id,
            'coords':receptor_coords, 
            'pocket_flag':receptor_pocket_flag,
            'atom_feature':receptor_atom_feature, 
            'residue_number':receptor_residue_number, 
            'bond_dict':bond_dict
        }

    def __getitem__(self,idx):
        #return self._get_patch(self.pdb_list[idx]['code5'])
        logger.info(f"TOUGH_M1_dataset getitem:{idx}")
        return self.data[idx]



class TOUGH_M1_pair_dataset(TOUGH_M1_dataset):
    def __init__(self, pdb_list, pos_pairs=None, neg_pairs=None, data_root="./TOUGH-M1/"):
        super().__init__(pdb_list, data_root)
        # filter pairs to those supported by pdbs
        if pos_pairs==None:
            pos_pairs = []
            with open(data_root+"TOUGH-M1_positive.list","r") as f:
                lines = f.readlines()
                for line in lines:
                    spl = line.split()
                    pos_pairs.append([spl[0],spl[1]])
        if neg_pairs==None:
            neg_pairs = []
            with open(data_root+"TOUGH-M1_negative.list","r") as f:
                lines = f.readlines()
                for line in lines:
                    spl = line.split()
                    neg_pairs.append([spl[0],spl[1]])
        self._pos_pairs = list(filter(lambda p: p[0] in self._pdb_map.keys() and p[1] in self._pdb_map.keys(), pos_pairs))
        self._neg_pairs = list(filter(lambda p: p[0] in self._pdb_map.keys() and p[1] in self._pdb_map.keys(), neg_pairs))
        logger.info('Dataset positive pairs: {:d}, negative pairs: {:d}'.format(len(self._pos_pairs), len(self._neg_pairs)))
        
    def __len__(self):
        return len(self._pos_pairs)+len(self._neg_pairs)
    
    def __getitem__(self,indices):
        logger.info(f"TOUGH_M1_pair_dataset getitem:{indices}")
        if type(indices) == list:
            pocket_data = []
            label = []
            for i in indices:
                if i < len(self._neg_pairs):
                    similar_flag = 0
                    pocket1, pocket2 = self._neg_pairs[i]
                else:
                    similar_flag = 1
                    pocket1, pocket2 = self._pos_pairs[i-len(self._neg_pairs)]
                data1 = super().__getitem__(self._pdb_map[pocket1])
                data2 = super().__getitem__(self._pdb_map[pocket2])

                pocket_data.append([data1,data2])
                label.append(similar_flag)
            return {'pocket_data':pocket_data, 'label':torch.from_numpy(np.array(label))}
                
        else:
            idx = indices
            if idx < len(self._neg_pairs):
                similar_flag = 0
                pocket1, pocket2 = self._neg_pairs[idx]
            else:
                similar_flag = 1
                pocket1, pocket2 = self._pos_pairs[idx-len(self._neg_pairs)]
            data1 = super().__getitem__(self._pdb_map[pocket1])
            data2 = super().__getitem__(self._pdb_map[pocket2])

            return {'pocket_data':[[data1,data2]], 'label':torch.from_numpy(np.array([similar_flag]))}

