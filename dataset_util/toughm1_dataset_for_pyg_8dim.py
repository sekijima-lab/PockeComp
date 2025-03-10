
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max
from torch_geometric.nn import radius, radius_graph
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from biopandas.pdb import PandasPdb
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from collections import defaultdict
import logging
import random
import os
import traceback
import pickle


logger = logging.getLogger(__name__)
#logging.basicConfig(encoding='utf-8', level=logging.INFO)

def recalculate_features(points, features, radius_value=12):
    """
    points: torch tensor of shape (N, 3) containing the coordinates of N points
    features: torch tensor of shape (N, C) containing C features for each point (values are either 0 or 1)
    radius_value: float, the radius to consider for the neighborhood
    
    Returns:
    new_features: torch tensor of shape (N, C) containing the recalculated feature for each point
    """
    # Get the indices of points within the radius
    edge_index = radius_graph(points, r=radius_value, max_num_neighbors=points.size(0))
    edge_index, _ = add_self_loops(edge_index)
    row, col = edge_index
    logger.debug(f"row={row},col={col}")
    new_features = torch.zeros_like(features, device=points.device)

    # Get neighbors for point i
    neighbors = col#[col != row]
    indices = row#[row != col]
    #neighbors = neighbors[neighbors != i]  # Exclude the point itself
    logger.debug(f"neighbors={neighbors}, indices={indices}")
    if len(neighbors) > 0:
        # Extract features of neighbors
        neighbors_features = features[neighbors,:]
        logger.debug(f"neighbors_features:{neighbors_features}")
        # Calculate weights based on the given formula
        #xi = points[i]
        ra = neighbors_features.float()  # Convert to float for calculation
        distances = torch.norm(points[neighbors] - points[indices], dim=1)
        logger.debug(f"distances:{distances}")
        weights = ra*(1 - torch.exp(-(ra / distances.unsqueeze(1))**12))
        #weights[ra == 0] = 0  # Set weight to 0 if ra (neighbors_features) is 0
        logger.debug(f"weights:{weights}")
        # Calculate weighted features
        weighted_features = weights.float()

        # Find the maximum weight feature
        max_weight_features, _ = scatter_max(weighted_features, indices, dim=0)
        logger.debug(f"max_weight_features:{max_weight_features}")
        new_features = max_weight_features
    
    return new_features

def process_tensors_with_chains(tensor: torch.Tensor, chain_ids: list[str]):
    """
    residとchain ID（文字列）の組み合わせを考慮して番号を振る
    
    Args:
        tensor_list: residのリスト
        chain_ids_list: chain IDのリスト（文字列のリスト）
    """

    current_id = 0
    
    normalized_tensor = torch.zeros_like(tensor, dtype=torch.long)
    prev_value = None
    prev_chain = None
    
    for i, (value, chain) in enumerate(zip(tensor, chain_ids)):
        value = value.item()
        
        # 最初の値、もしくは前の値/chainと異なる場合は新しいIDを割り当て
        if (prev_value is None or 
            prev_chain is None or 
            value != prev_value or 
            chain != prev_chain):
            current_id += 1
        
        normalized_tensor[i] = current_id - 1
        prev_value = value
        prev_chain = chain
        
    
    return normalized_tensor

class TOUGH_M1_dataset(Dataset):
    def __init__(self,pdb_list,data_root="./TOUGH-M1/",recalculate=False,p2rank_prediction_apply=True,col=' probability'):
        logger.debug(f"TOUGH_M1_dataset initialize {self.__class__.__name__}")
        self.data_root = data_root
        self.pdb_list = pdb_list
        self.recalculate = recalculate
        self.p2rank_prediction_apply = p2rank_prediction_apply

        # map pdb code -> entry
        self._pdb_map = {}
        self._idx_map = {}
        assert col in [' probability', ' score', ' zscore']
        self.col = col
        self.data, self.p2rank_info = self.preprocess()
        
        for i, pdb_entry in enumerate(self.data):
            code = pdb_entry['pocket_id']
            self._pdb_map[code] = i
            self._idx_map[i] = code
            
    def __len__(self):
        return len(self.data)

    def set_col(self,col):
        assert col in [' probability', ' score', ' zscore']
        if self.col == col or self.p2rank_prediction_apply == False:
            return
        logger.info(f"p2rank_col {self.col} -> {col}")
        self.col = col
        for i in tqdm(range(len(self.data))):
            self.data[i]['p2rank_pocket_prediction'] = self.p2rank_info[i][col]

    def set_recalculate(self,recalculate=False, device='cpu'):
        if self.recalculate == recalculate:
            return
        if recalculate == True:
            logger.info(f"recalculate {self.recalculate} -> {recalculate}")
            self.recalculate = recalculate
            for i in tqdm(range(len(self.data))):
                self.data[i]['atom_feature'] = recalculate_features(self.data[i]['coords'].to(device),self.data[i]['atom_feature'].to(device)).cpu()
        else:
            raise ValueError("cannot set recalculate True to False")
            
    def preprocess(self):
        logger.debug("TOUGH_M1_dataset preprocessing")
        data = []
        p2rank_info = []
        for pdb_entry in tqdm(self.pdb_list):
            try:
                obj, p2rank_predictions = self.get_data_per_protein(pdb_entry)
            except:
                logger.info("{} is failed to get data:{}".format(pdb_entry["code5"],traceback.format_exc()))
                continue
            data.append(obj)
            p2rank_info.append(p2rank_predictions)
        logger.info("TOUGH_M1_dataset preprocess done")
        return data, p2rank_info
    
    def p2rank_data_extract(self, entry):
        if not os.path.exists(entry["p2rank_prediction"]):
            raise FileNotFoundError("Not Found '{}'".format(entry["p2rank_prediction"]))
        df = pd.read_csv(entry["p2rank_prediction"])
        dict_file_path = entry["p2rank_prediction"].replace(".csv",".pkl")
        try:
            if os.path.exists(dict_file_path):
                try:
                    with open(dict_file_path, 'rb') as f:
                        d = pickle.load(f)
                    for col in [' probability', ' score', ' zscore']:
                        assert type(d[col]) == defaultdict
                except pickle.UnpicklingError:
                    raise ValueError(f"'{dict_file_path}' is an invalid pickle file.")
            else:
                raise ValueError(f"'{dict_file_path}' is a non-existent pickle file.")
        except:
            d_proba = defaultdict(float)
            d_score = defaultdict(float)
            d_zscore = defaultdict(float)
            for chain, residue_number, probability, score, zscore in zip(df['chain'],df[' residue_label'], df[' probability'], df[' score'], df[' zscore']):
                try:
                    d_proba[(chain.strip(), int(residue_number))] = probability
                    d_score[(chain.strip(), int(residue_number))] = score
                    d_zscore[(chain.strip(), int(residue_number))] = zscore
                except:
                    logger.warning(f"{traceback.format_exc()}, failed to set dict key:({chain},{residue_number})")
            d = {' probability':d_proba, ' score':d_score, ' zscore':d_zscore}
            with open(dict_file_path, 'wb') as f:
                pickle.dump(d, f)
            logger.debug(f"saved to 'defaultdict' object to '{dict_file_path}'")
            
        #print(df.columns)
        return d
    
    def get_data_per_protein(self,entry):
        code = entry['code5'] if 'code5' in entry else entry['code']
        logger.debug(f"Get data of protein id:{code}")
        #receptor= PandasPdb().read_pdb(dir+f'/{id}.pdb')
        #ligand = PandasPdb().read_pdb(dir+f'/{id}00.pdb')
        
        """
        f_text = subprocess.run("obabel -ipdb "+entry["protein"]+" -omol2", shell=True, stdout=subprocess.PIPE , stderr=subprocess.PIPE ,encoding="utf-8").stdout
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
        """
        # 座標
        #receptor_coords = receptor.df['ATOM'][['x_coord','y_coord','z_coord']].to_numpy()
        #ligand_coords = ligand.df['HETATM'][['x_coord','y_coord','z_coord']].to_numpy()
    
        # 距離行列からポケット部位抽出
        #dist_mat = distance_matrix(ligand_coords,receptor_coords)
        #dist_atom_to_ligand = np.min(dist_mat,axis=0)
        #pocket_residue_numbers = list(receptor.df['ATOM'][dist_atom_to_ligand < 8]['residue_number'].unique())

        #def pocket_flag(x):
        #    return 1 if x in pocket_residue_numbers else 0
        #receptor.df['ATOM']['pocket_flag'] = receptor.df['ATOM']['residue_number'].apply(pocket_flag)
    
        # 結合情報
        """
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
        """
        bond_dict = None
        # 特徴量抽出
        #receptor_pocket_flag = receptor.df['ATOM']['pocket_flag'].to_numpy().astype(np.float64)
        #receptor_atom_feature = np.array([
        #    receptor.df['ATOM']['element_symbol']=='N',
        #    receptor.df['ATOM']['element_symbol']=='C',
        #    receptor.df['ATOM']['element_symbol']=='O',
        #    receptor.df['ATOM']['element_symbol']=='S',
        #]).astype(np.float64).T
        #receptor_residue_number = receptor.df['ATOM']['residue_number'].to_numpy()
        precomputed_features_dict = torch.load(entry["moleculekit_features"])
        receptor_coords = precomputed_features_dict["coords"]
        receptor_pocket_flag = precomputed_features_dict["pocket_flag"]
        receptor_atom_feature = precomputed_features_dict["channels"]
        receptor_residue_number = precomputed_features_dict["residue_number"]

        if self.recalculate:
            receptor_atom_feature = recalculate_features(receptor_coords,receptor_atom_feature)

        try:
            receptor_chain_ids = precomputed_features_dict["chains"]
        except:
            logger.warning(f"cannot load chain id info of {code}, treat it as if it were a single chain protein.")
            receptor_chain_ids = torch.ones_like(receptor_residue_number, dtype=torch.long)
            

        p2rank_pocket_prediction = None
        p2rank_pocket_predictions = None
        if self.p2rank_prediction_apply:
            dicts = self.p2rank_data_extract(entry)
            #p2rank_pocket_predictions = {
                # key: torch.tensor([
                    # d[(receptor_chain_ids[i], receptor_residue_number.tolist()[i])] 
                    # for i in range(len(receptor_residue_number))
                    # ]) 
                    # for key,d in dicts.items()
                # }
            p2rank_pocket_predictions = {
                key: torch.tensor([
                    d[(receptor_chain_ids[i], receptor_residue_number.tolist()[i])]  # これはfloat値を直接返す
                    for i in range(len(receptor_residue_number))
                ]) 
                for key, d in dicts.items()
            }
            p2rank_pocket_prediction = p2rank_pocket_predictions[self.col]
        # torch.tensor化
        #receptor_coords = torch.from_numpy(receptor_coords).float()
        #receptor_pocket_flag = torch.from_numpy(receptor_pocket_flag).float()
        #receptor_atom_feature = torch.from_numpy(receptor_atom_feature).float()
        #receptor_residue_number = torch.from_numpy(receptor_residue_number)

        if len(set(receptor_chain_ids)) > 1: # receptor_chain_idsはただのリスト
            receptor_residue_number = process_tensors_with_chains(receptor_residue_number,receptor_chain_ids)
        #receptor_residue_number = reindex_monotonic(receptor_residue_number)
        return {
            'pocket_id':code,
            'coords':receptor_coords, 
            'pocket_flag':receptor_pocket_flag,
            'atom_feature':receptor_atom_feature,
            'residue_number':receptor_residue_number, 
            'bond_dict':bond_dict,
            'p2rank_pocket_prediction': p2rank_pocket_prediction
        }, p2rank_pocket_predictions

    def __getitem__(self,idx):
        #return self._get_patch(self.pdb_list[idx]['code5'])
        logger.debug(f"TOUGH_M1_dataset getitem:{idx}")
        return self.data[idx]
    
def reindex_monotonic(indices):
    # indicesをテンソルに変換（まだテンソルでない場合）
    indices = torch.tensor(indices) if not isinstance(indices, torch.Tensor) else indices

    # ユニークな値を取得し、元の順序を保持
    unique_indices, inverse_indices = torch.unique(indices, sorted=True, return_inverse=True)

    # 0から始まる新しいインデックスを作成
    new_indices = torch.arange(len(unique_indices))

    # 元のインデックスを新しいインデックスに置き換え
    reindexed = new_indices[inverse_indices]

    return reindexed
    
class TOUGH_M1_segmentation_dataset(TOUGH_M1_dataset):
    def __init__(self,pdb_list,data_root="./TOUGH-M1/",recalculate=False):
        super().__init__(pdb_list, data_root,recalculate,p2rank_prediction_apply=False)
        for i in range(len(self.data)):
            receptor_residue_number = self.data[i]["residue_number"]
            self.data[i]["residue_number"] = reindex_monotonic(receptor_residue_number)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        pocket_id, coords, pocket_flag, atom_feature, receptor_residue_number, _, _ = d.values()
        
        return Data(
            x = atom_feature,
            pos = coords,
            pocket_flag = pocket_flag,
            batch_res = receptor_residue_number,
            code5=pocket_id
        )

class TOUGH_M1_pocket_dataset(TOUGH_M1_dataset):
    def __init__(self,pdb_list,data_root="./TOUGH-M1/",recalculate=False,**kwargs):
        super().__init__(pdb_list, data_root,recalculate,**kwargs)
        for i in range(len(self.data)):
            receptor_residue_number = self.data[i]["residue_number"]
            pocket_flag = self.data[i]["pocket_flag"]
            self.data[i]["residue_number"] = reindex_monotonic(receptor_residue_number[pocket_flag==1])
        
        
    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        pocket_id, coords, pocket_flag, atom_feature, receptor_residue_number, _, p2rank_pocket_prediction = d.values()
        #logger.debug(f"coords.size()={coords.size()},pocket_flag.size()={pocket_flag.size()},atom_feature.size()={atom_feature.size()}")
        new_coords = coords[pocket_flag==1,:]
        new_atom_feature = atom_feature[pocket_flag==1,:]
        new_residue_number = receptor_residue_number
        if self.p2rank_prediction_apply:
            new_pocket_flag = p2rank_pocket_prediction[pocket_flag==1]
        else:
            new_pocket_flag = pocket_flag[pocket_flag==1]
        
        return Data(
            x=new_atom_feature,
            pos=new_coords,
            batch_res=new_residue_number,
            code5=pocket_id,
            pocket_flag=new_pocket_flag
        )
            


class TOUGH_M1_pair_dataset(TOUGH_M1_pocket_dataset):
    def __init__(self, pdb_list, pos_pairs=None, neg_pairs=None, data_root="./TOUGH-M1/",recalculate=False,**kwargs):
        super().__init__(pdb_list, data_root,recalculate,**kwargs)
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
        
    def indices_to_pocket_code(self,indices):
        logger.debug(f"indices:{indices}")
        assert len(indices)%2 == 0
        pocket_ids = []
        for i in range(len(indices)//2):
            tmp = []
            for j in range(2):
                tmp.append(self._idx_map[indices[2*i+j].item()])
            pocket_ids.append(tmp)
        return pocket_ids
    
    def __len__(self):
        #return len(self._pos_pairs)+len(self._neg_pairs)
        return len(self._pos_pairs) * 2
    
    def __getitem__(self,indices):
        logger.debug(f"TOUGH_M1_pair_dataset getitem:{indices}")
        idx = indices
        if idx >= len(self._pos_pairs):
            similar_flag = 0
            pocket1, pocket2 = random.choice(self._neg_pairs)
        else:
            similar_flag = 1
            pocket1, pocket2 = self._pos_pairs[idx]
        #Data object, x=(n,m), pos=(n,3), id
        data1 = super().__getitem__(self._pdb_map[pocket1])
        data2 = super().__getitem__(self._pdb_map[pocket2])
        

        return Data(
            x=torch.cat([data1.x,data2.x],dim=0),
            pos=torch.cat([data1.pos,data2.pos],dim=0),
            code5=torch.tensor([self._pdb_map[pocket1],self._pdb_map[pocket2]]),
            batch_res=process_tensors_monotonically([data1.batch_res,data2.batch_res]),
            pocket_flag=torch.cat([data1.pocket_flag, data2.pocket_flag],dim=0),
            batch_sub=torch.tensor([0 for i in range(data1.x.size()[0])]+[1 for i in range(data2.x.size()[0])]),
            label=torch.tensor(np.array([similar_flag]))
            )

class TOUGH_M1_pair_dataset_sample(TOUGH_M1_pair_dataset):
    def __init__(self, pdb_list, length=1000, pos_pairs=None, neg_pairs=None, data_root="./TOUGH-M1/",recalculate=False):
        super().__init__(pdb_list, pos_pairs, neg_pairs, data_root,recalculate)
        self._pos_pairs = random.sample(self._pos_pairs, min(length//2,len(self._pos_pairs)))
        self._neg_pairs = random.sample(self._neg_pairs, min(length - length//2, len(self._neg_pairs)))
        logger.info('Sampled Dataset positive pairs: {:d}, negative pairs: {:d}'.format(len(self._pos_pairs), len(self._neg_pairs)))
        
class TOUGH_M1_pair_dataset_eval(TOUGH_M1_pair_dataset):
    def __init__(self, pdb_list, pos_pairs=None, neg_pairs=None, data_root="./TOUGH-M1/",recalculate=False,**kwargs):
        super().__init__(pdb_list, pos_pairs, neg_pairs, data_root, recalculate,**kwargs)
        
    def __len__(self):
        return len(self._pos_pairs) + len(self._neg_pairs)
    
    def __getitem__(self,indices):
        logger.debug(f"TOUGH_M1_pair_dataset getitem:{indices}")
        idx = indices
        if idx >= len(self._pos_pairs):
            similar_flag = 0
            pocket1, pocket2 = self._neg_pairs[idx-len(self._pos_pairs)]
        else:
            similar_flag = 1
            pocket1, pocket2 = self._pos_pairs[idx]
        #Data object, x=(n,m), pos=(n,3), id
        data1 = super().__getitem__(self._pdb_map[pocket1])
        data2 = super().__getitem__(self._pdb_map[pocket2])
        

        return Data(
            x=torch.cat([data1.x,data2.x],dim=0),
            pos=torch.cat([data1.pos,data2.pos],dim=0),
            code5=torch.tensor([self._pdb_map[pocket1],self._pdb_map[pocket2]]),
            batch_res=process_tensors_monotonically([data1.batch_res,data2.batch_res]),
            pocket_flag=torch.cat([data1.pocket_flag, data2.pocket_flag],dim=0),
            batch_sub=torch.tensor([0 for i in range(data1.x.size()[0])]+[1 for i in range(data2.x.size()[0])]),
            label=torch.tensor(np.array([similar_flag]))
            )
        
class TOUGH_M1_triplet_dataset(TOUGH_M1_pair_dataset):
    def __init__(self, pdb_list, pos_pairs=None, neg_pairs=None, data_root="./TOUGH-M1/", recalculate=False,**kwargs):
        super().__init__(pdb_list, pos_pairs, neg_pairs, data_root, recalculate,**kwargs)
        positive_dict = dict()
        negative_dict = dict()
        for pair in self._pos_pairs:
            pocket1, pocket2 = pair
            if pocket1 not in positive_dict.keys():
                positive_dict[pocket1] = [pocket2]
            else:
                positive_dict[pocket1].append(pocket2)
            if pocket2 not in positive_dict.keys():
                positive_dict[pocket2] = [pocket1]
            else:
                positive_dict[pocket2].append(pocket1)
        for pair in self._neg_pairs:
            pocket1, pocket2 = pair
            if pocket1 not in negative_dict.keys():
                negative_dict[pocket1] = [pocket2]
            else:
                negative_dict[pocket1].append(pocket2)
            if pocket2 not in negative_dict.keys():
                negative_dict[pocket2] = [pocket1]
            else:
                negative_dict[pocket2].append(pocket1)

        self.positive_dict = positive_dict
        self.negative_dict = negative_dict
        self.effective_pdb_list = list(set(positive_dict.keys()) | set(negative_dict.keys()))
        logger.debug(self.data)
        tmp_data = [(entry,info) for entry,info in zip(self.data,self.p2rank_info) if entry['pocket_id'] in self.effective_pdb_list]
        self.data = [d[0] for d in tmp_data]
        self.p2rank_info = [d[1] for d in tmp_data]
        self.effective_code_map = {self.effective_pdb_list[i]:i for i in range(len(self.effective_pdb_list))}
        self.effective_pdb_list_to_idx = {}
        for i,entry in enumerate(self.data):
            try:
                effective_idx = self.effective_pdb_list.index(entry['pocket_id'])
                self.effective_pdb_list_to_idx[effective_idx] = i
            except:
                continue 
        logger.info(f"Effective pdb files : {len(self.effective_pdb_list)}")
    
    def __len__(self):
        return len(self.effective_pdb_list)
    
    def __getitem__(self, indices):
        if type(indices)==list:
            indices = [self.effective_pdb_list_to_idx[effective_idx] for effective_idx in indices]
        else:
            indices = self.effective_pdb_list_to_idx[indices]
        return TOUGH_M1_pocket_dataset.__getitem__(self,indices)
    
    def get_triplet_info(self, idx):
        code = self.effective_pdb_list[idx]
        #data_main = TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code])
        positive_list = self.positive_dict.get(code,[])
        positive_list = torch.tensor([self.effective_code_map[code] for code in positive_list])
        #data_positive = [TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code]) for code  in positive_list]
        negative_list = self.negative_dict.get(code,[])
        negative_list = torch.tensor([self.effective_code_map[code] for code in negative_list])
        #data_negative = [TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code]) for code  in positive_list]
        return {
            "code":code,
            "positive_list":positive_list,
            "negative_list":negative_list
        }
        
    def get_effective_pdb_list(self):
        return self.effective_pdb_list
    

class VERTEX_pair_dataset(TOUGH_M1_pocket_dataset):
    def __init__(self, pdb_list, data_root="./Vertex/",recalculate=False,**kwargs):
        super().__init__(pdb_list, data_root,recalculate,**kwargs)
        # filter pairs to those supported by pdbs
        self.pos_pairs = []
        self.neg_pairs = []
        with open(os.path.join(self.data_root, 'protein_pairs.tsv')) as f:
            for i, line in enumerate(f.readlines()):
                if i > 1:
                    tokens = line.split('\t')
                    pdb1, pdb2, cls = tokens[0].lower(), tokens[5].lower(), int(tokens[-1])
                    if pdb1 in self._pdb_map.keys() and pdb2 in self._pdb_map.keys():
                        if cls == 1:
                            self.pos_pairs.append([pdb1,pdb2])
                        else:
                            self.neg_pairs.append([pdb1,pdb2])
        self._pos_pairs = list(filter(lambda p: p[0] in self._pdb_map.keys() and p[1] in self._pdb_map.keys(), self.pos_pairs))
        self._neg_pairs = list(filter(lambda p: p[0] in self._pdb_map.keys() and p[1] in self._pdb_map.keys(), self.neg_pairs))
        logger.info('Dataset positive pairs: {:d}, negative pairs: {:d}'.format(len(self._pos_pairs), len(self._neg_pairs)))
    
    def __len__(self):
        #return len(self._pos_pairs)+len(self._neg_pairs)
        return len(self._pos_pairs) * 2
    
    def __getitem__(self,indices):
        logger.debug(f"{self.__class__} getitem:{indices}")
        idx = indices
        if idx >= len(self._pos_pairs):
            similar_flag = 0
            pocket1, pocket2 = random.choice(self._neg_pairs)
        else:
            similar_flag = 1
            pocket1, pocket2 = self._pos_pairs[idx]
        #Data object, x=(n,m), pos=(n,3), id
        data1 = super().__getitem__(self._pdb_map[pocket1])
        data2 = super().__getitem__(self._pdb_map[pocket2])
        

        return Data(
            x=torch.cat([data1.x,data2.x],dim=0),
            pos=torch.cat([data1.pos,data2.pos],dim=0),
            code5=torch.tensor([self._pdb_map[pocket1],self._pdb_map[pocket2]]),
            batch_res=process_tensors_monotonically([data1.batch_res,data2.batch_res]),
            pocket_flag=torch.cat([data1.pocket_flag, data2.pocket_flag],dim=0),
            batch_sub=torch.tensor([0 for i in range(data1.x.size()[0])]+[1 for i in range(data2.x.size()[0])]),
            label=torch.tensor(np.array([similar_flag]))
            )
        
class VERTEX_pair_dataset_eval(VERTEX_pair_dataset):
    def __init__(self, pdb_list, data_root="./Vertex/",recalculate=False,**kwargs):
        super().__init__(pdb_list, data_root, recalculate,**kwargs)
        
    def __len__(self):
        return len(self._pos_pairs) + len(self._neg_pairs)
    
    def __getitem__(self,indices):
        logger.debug(f"{self.__class__} getitem:{indices}")
        idx = indices
        if idx >= len(self._pos_pairs):
            similar_flag = 0
            pocket1, pocket2 = self._neg_pairs[idx-len(self._pos_pairs)]
        else:
            similar_flag = 1
            pocket1, pocket2 = self._pos_pairs[idx]
        #Data object, x=(n,m), pos=(n,3), id
        data1 = super().__getitem__(self._pdb_map[pocket1])
        data2 = super().__getitem__(self._pdb_map[pocket2])
        

        return Data(
            x=torch.cat([data1.x,data2.x],dim=0),
            pos=torch.cat([data1.pos,data2.pos],dim=0),
            code5=torch.tensor([self._pdb_map[pocket1],self._pdb_map[pocket2]]),
            batch_res=process_tensors_monotonically([data1.batch_res,data2.batch_res]),
            pocket_flag=torch.cat([data1.pocket_flag, data2.pocket_flag],dim=0),
            batch_sub=torch.tensor([0 for i in range(data1.x.size()[0])]+[1 for i in range(data2.x.size()[0])]),
            label=torch.tensor(np.array([similar_flag]))
            )

class VERTEX_triplet_dataset(VERTEX_pair_dataset):
    def __init__(self, pdb_list, data_root="./Vertex/", recalculate=False,**kwargs):
        super().__init__(pdb_list, data_root, recalculate,**kwargs)
        positive_dict = dict()
        negative_dict = dict()
        for pair in self._pos_pairs:
            pocket1, pocket2 = pair
            if pocket1 not in positive_dict.keys():
                positive_dict[pocket1] = [pocket2]
            else:
                positive_dict[pocket1].append(pocket2)
            if pocket2 not in positive_dict.keys():
                positive_dict[pocket2] = [pocket1]
            else:
                positive_dict[pocket2].append(pocket1)
        for pair in self._neg_pairs:
            pocket1, pocket2 = pair
            if pocket1 not in negative_dict.keys():
                negative_dict[pocket1] = [pocket2]
            else:
                negative_dict[pocket1].append(pocket2)
            if pocket2 not in negative_dict.keys():
                negative_dict[pocket2] = [pocket1]
            else:
                negative_dict[pocket2].append(pocket1)

        self.positive_dict = positive_dict
        self.negative_dict = negative_dict
        self.effective_pdb_list = list(set(positive_dict.keys()) | set(negative_dict.keys()))
        logger.debug(self.data)
        tmp_data = [(entry,info) for entry,info in zip(self.data,self.p2rank_info) if entry['pocket_id'] in self.effective_pdb_list]
        self.data = [d[0] for d in tmp_data]
        self.p2rank_info = [d[1] for d in tmp_data]
        self.effective_code_map = {self.effective_pdb_list[i]:i for i in range(len(self.effective_pdb_list))}
        self.effective_pdb_list_to_idx = {}
        for i,entry in enumerate(self.data):
            try:
                effective_idx = self.effective_pdb_list.index(entry['pocket_id'])
                self.effective_pdb_list_to_idx[effective_idx] = i
            except:
                continue 
        logger.info(f"Effective pdb files : {len(self.effective_pdb_list)}")
    
    def __len__(self):
        return len(self.effective_pdb_list)
    
    def __getitem__(self, indices):
        if type(indices)==list:
            indices = [self.effective_pdb_list_to_idx[effective_idx] for effective_idx in indices]
        else:
            indices = self.effective_pdb_list_to_idx[indices]
        return TOUGH_M1_pocket_dataset.__getitem__(self,indices)
    
    def get_triplet_info(self, idx):
        code = self.effective_pdb_list[idx]
        #data_main = TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code])
        positive_list = self.positive_dict.get(code,[])
        positive_list = torch.tensor([self.effective_code_map[code] for code in positive_list])
        #data_positive = [TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code]) for code  in positive_list]
        negative_list = self.negative_dict.get(code,[])
        negative_list = torch.tensor([self.effective_code_map[code] for code in negative_list])
        #data_negative = [TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code]) for code  in positive_list]
        return {
            "code":code,
            "positive_list":positive_list,
            "negative_list":negative_list
        }
        
    def get_effective_pdb_list(self):
        return self.effective_pdb_list
    
class PROSPECCTS_pair_dataset(TOUGH_M1_pocket_dataset):
    def __init__(self, pdb_list, dbname="P1", data_root="./prospeccts/",recalculate=False,**kwargs):
        super().__init__(pdb_list, data_root, recalculate,**kwargs)
        logger.debug(f"self._pdb_map:{self._pdb_map}")
        self.dbname = dbname
        dir1, dir2, listfn = self._prospeccts_paths()
        root = os.path.join(self.data_root, dir1)
        self.pos_pairs = []
        self.neg_pairs = []
        
        with open(os.path.join(root, listfn)) as f:
            logger.debug(f"Opening file {os.path.join(root, listfn)}")
            for line in f.readlines():
                tokens = line.split(',')
                id1, id2, cls = tokens[0], tokens[1], tokens[2].strip()
                logger.debug(f"id1:{id1,id1 in self._pdb_map.keys()},id2:{id2,id2 in self._pdb_map.keys()}")
                if id1 in self._pdb_map.keys() and id2 in self._pdb_map.keys():
                    if cls=="active":
                        self.pos_pairs.append([id1,id2])
                    else:
                        self.neg_pairs.append([id1,id2])
                        
        self._pos_pairs = list(filter(lambda p: p[0] in self._pdb_map.keys() and p[1] in self._pdb_map.keys(), self.pos_pairs))
        self._neg_pairs = list(filter(lambda p: p[0] in self._pdb_map.keys() and p[1] in self._pdb_map.keys(), self.neg_pairs))
        logger.info('Dataset positive pairs: {:d}, negative pairs: {:d}'.format(len(self._pos_pairs), len(self._neg_pairs)))
        
    def _prospeccts_paths(self):
        if self.dbname == 'P1':
            dir1, dir2, listfn = 'identical_structures', 'identical_structures', 'identical_structures.csv'
        elif self.dbname == 'P1.2':
            dir1, dir2, listfn = 'identical_structures_similar_ligands', 'identical_structures_similar_ligands', 'identical_structures_similar_ligands.csv'
        elif self.dbname == 'P2':
            dir1, dir2, listfn = 'NMR_structures', 'NMR_structures', 'NMR_structures.csv'
        elif self.dbname == 'P3':
            dir1, dir2, listfn = 'decoy', 'decoy_structures', 'decoy_structures5.csv'
        elif self.dbname == 'P4':
            dir1, dir2, listfn = 'decoy', 'decoy_shape_structures', 'decoy_structures5.csv'
        elif self.dbname == 'P5':
            dir1, dir2, listfn = 'kahraman_structures', 'kahraman_structures', 'kahraman_structures80.csv'
        elif self.dbname == 'P5.2':
            dir1, dir2, listfn = 'kahraman_structures', 'kahraman_structures', 'kahraman_structures.csv'
        elif self.dbname == 'P6':
            dir1, dir2, listfn = 'barelier_structures', 'barelier_structures', 'barelier_structures.csv'
        elif self.dbname == 'P6.2':
            dir1, dir2, listfn = 'barelier_structures', 'barelier_structures_cofactors', 'barelier_structures.csv'
        elif self.dbname == 'P7':
            dir1, dir2, listfn = 'review_structures', 'review_structures', 'review_structures.csv'
        else:
            raise NotImplementedError
        logger.debug(f"path data of {self.dbname} is {dir1, dir2, listfn}")
        return dir1, dir2, listfn
    
    def __len__(self):
        #return len(self._pos_pairs)+len(self._neg_pairs)
        return len(self._pos_pairs) * 2
    
    def __getitem__(self,indices):
        logger.debug(f"{self.__class__} getitem:{indices}")
        idx = indices
        if idx >= len(self._pos_pairs):
            similar_flag = 0
            pocket1, pocket2 = random.choice(self._neg_pairs)
        else:
            similar_flag = 1
            pocket1, pocket2 = self._pos_pairs[idx]
        #Data object, x=(n,m), pos=(n,3), id
        data1 = super().__getitem__(self._pdb_map[pocket1])
        data2 = super().__getitem__(self._pdb_map[pocket2])
        

        return Data(
            x=torch.cat([data1.x,data2.x],dim=0),
            pos=torch.cat([data1.pos,data2.pos],dim=0),
            code5=torch.tensor([self._pdb_map[pocket1],self._pdb_map[pocket2]]),
            batch_res=process_tensors_monotonically([data1.batch_res,data2.batch_res]),
            pocket_flag=torch.cat([data1.pocket_flag, data2.pocket_flag],dim=0),
            batch_sub=torch.tensor([0 for i in range(data1.x.size()[0])]+[1 for i in range(data2.x.size()[0])]),
            label=torch.tensor(np.array([similar_flag]))
            )

class PROSPECCTS_pair_dataset_eval(PROSPECCTS_pair_dataset):
    def __init__(self, pdb_list, dbname="P1", pos_pairs=None, neg_pairs=None, data_root="./prospeccts/",recalculate=False,**kwargs):
        super().__init__(pdb_list, dbname, data_root, recalculate,**kwargs)
        
    def __len__(self):
        return len(self._pos_pairs) + len(self._neg_pairs)
    
    def __getitem__(self,indices):
        logger.debug(f"{self.__class__} getitem:{indices}")
        idx = indices
        if idx >= len(self._pos_pairs):
            similar_flag = 0
            pocket1, pocket2 = self._neg_pairs[idx-len(self._pos_pairs)]
        else:
            similar_flag = 1
            pocket1, pocket2 = self._pos_pairs[idx]
        #Data object, x=(n,m), pos=(n,3), id
        data1 = super().__getitem__(self._pdb_map[pocket1])
        data2 = super().__getitem__(self._pdb_map[pocket2])
        

        return Data(
            x=torch.cat([data1.x,data2.x],dim=0),
            pos=torch.cat([data1.pos,data2.pos],dim=0),
            code5=torch.tensor([self._pdb_map[pocket1],self._pdb_map[pocket2]]),
            batch_res=process_tensors_monotonically([data1.batch_res,data2.batch_res]),
            pocket_flag=torch.cat([data1.pocket_flag, data2.pocket_flag],dim=0),
            batch_sub=torch.tensor([0 for i in range(data1.x.size()[0])]+[1 for i in range(data2.x.size()[0])]),
            label=torch.tensor(np.array([similar_flag]))
            )        
        
class PROSPECCTS_triplet_dataset(PROSPECCTS_pair_dataset):
    def __init__(self, pdb_list, dbname="P1", data_root="./prospeccts/",recalculate=False,**kwargs):
        super().__init__(pdb_list, dbname, data_root, recalculate,**kwargs)
        positive_dict = dict()
        negative_dict = dict()
        for pair in self._pos_pairs:
            pocket1, pocket2 = pair
            if pocket1 not in positive_dict.keys():
                positive_dict[pocket1] = [pocket2]
            else:
                positive_dict[pocket1].append(pocket2)
            if pocket2 not in positive_dict.keys():
                positive_dict[pocket2] = [pocket1]
            else:
                positive_dict[pocket2].append(pocket1)
        for pair in self._neg_pairs:
            pocket1, pocket2 = pair
            if pocket1 not in negative_dict.keys():
                negative_dict[pocket1] = [pocket2]
            else:
                negative_dict[pocket1].append(pocket2)
            if pocket2 not in negative_dict.keys():
                negative_dict[pocket2] = [pocket1]
            else:
                negative_dict[pocket2].append(pocket1)

        self.positive_dict = positive_dict
        self.negative_dict = negative_dict
        self.effective_pdb_list = list(set(positive_dict.keys()) | set(negative_dict.keys()))
        logger.debug(self.data)
        tmp_data = [(entry,info) for entry,info in zip(self.data,self.p2rank_info) if entry['pocket_id'] in self.effective_pdb_list]
        self.data = [d[0] for d in tmp_data]
        self.p2rank_info = [d[1] for d in tmp_data]
        self.effective_code_map = {self.effective_pdb_list[i]:i for i in range(len(self.effective_pdb_list))}
        self.effective_pdb_list_to_idx = {}
        for i,entry in enumerate(self.data):
            try:
                effective_idx = self.effective_pdb_list.index(entry['pocket_id'])
                self.effective_pdb_list_to_idx[effective_idx] = i
            except:
                continue 
        logger.info(f"Effective pdb files : {len(self.effective_pdb_list)}")
    
    def __len__(self):
        return len(self.effective_pdb_list)
    
    def __getitem__(self, indices):
        if type(indices)==list:
            indices = [self.effective_pdb_list_to_idx[effective_idx] for effective_idx in indices]
        else:
            indices = self.effective_pdb_list_to_idx[indices]
        return TOUGH_M1_pocket_dataset.__getitem__(self,indices)
    
    def get_triplet_info(self, idx):
        code = self.effective_pdb_list[idx]
        #data_main = TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code])
        positive_list = self.positive_dict.get(code,[])
        positive_list = torch.tensor([self.effective_code_map[code] for code in positive_list])
        #data_positive = [TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code]) for code  in positive_list]
        negative_list = self.negative_dict.get(code,[])
        negative_list = torch.tensor([self.effective_code_map[code] for code in negative_list])
        #data_negative = [TOUGH_M1_pocket_dataset.__getitem__(self,self._pdb_map[code]) for code  in positive_list]
        return {
            "code":code,
            "positive_list":positive_list,
            "negative_list":negative_list
        }
        
    def get_effective_pdb_list(self):
        return self.effective_pdb_list

        

def process_tensors_monotonically(tensor_list):
    # 条件に合う値の抽出
    
    # 0始まりに振り直し、かつ連結後も連続するように調整
    normalized_tensors = []
    offset = 0
    for tensor in tensor_list:
        unique_values, inverse_indices = torch.unique(tensor, return_inverse=True)
        normalized_tensor = inverse_indices + offset
        normalized_tensors.append(normalized_tensor)
        offset += len(unique_values)
    
    # 連結
    concatenated_tensor = torch.cat(normalized_tensors)
    
    return concatenated_tensor

def TOUGH_M1_pair_dataset_collate_fn(data_list : list[Data]):
    batch_x = [data.x for data in data_list]
    batch_pos = [data.pos for data in data_list]
    batch_label = [data.label for data in data_list]
    batch_indices = [data.batch_sub + 2*i for i,data in enumerate(data_list)]
    batch_ids = [data.id for data in data_list]
    batch_res = [data.batch_res for data in data_list]
    batch_res_monotonically = process_tensors_monotonically(batch_res)
    batch_pocket_flags = [data.pocket_flag for data in data_list]
    # 新しい Data オブジェクトを作成する
    batch_data = Data(
        x=torch.cat(batch_x), 
        pos=torch.cat(batch_pos), 
        code5=torch.cat(batch_ids),
        batch_res=batch_res_monotonically,
        pocket_flag=torch.cat(batch_pocket_flags),
        label=torch.cat(batch_label), 
        batch_sub=torch.cat(batch_indices)
        )

    return batch_data