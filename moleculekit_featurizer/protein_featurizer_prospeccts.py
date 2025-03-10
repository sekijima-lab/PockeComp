import moleculekit
from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping, metal_atypes
from moleculekit.tools.voxeldescriptors import getChannels
import torch
import sys
import glob
import os
import logging
import re
from time import time
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance_matrix
import traceback
#from multiprocessing import Pool,cpu_count
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import shutil
# 特徴量抽出のみ

logger = logging.getLogger(__name__)
dist = 8
SUCCESS_COUNT = 0


def search_pdb_files(directory):
    logger.debug(directory)
    matching_files = glob.glob(os.path.join(directory,"**/**/*_clean.pdb"))
    return matching_files

def convert_chain_ids_to_integers(chain_ids: np.ndarray) -> torch.Tensor:
    """
    文字列の配列として与えられるchain IDsを整数のtensorに変換する
    """
    # ユニークなchain IDを取得し、各IDに整数を割り当てる
    unique_chains = np.unique(chain_ids)
    chain_to_int = {chain: idx for idx, chain in enumerate(unique_chains)}
    
    # 整数配列に変換
    int_chains = np.array([chain_to_int[chain] for chain in chain_ids])
    
    return torch.tensor(int_chains, dtype=torch.long)

def default_featurizer(pdb_path):
    try:
        prot = Molecule(pdb_path)
        
        mol_path = glob.glob(pdb_path.replace("_clean.pdb","_lig_*.pdb"))[0]
        mol = Molecule(mol_path)
        #bonds = prot.bonds #[n,2]
        #bonds = torch.from_numpy(bonds).squeeze()
        
        prot_init = prot.copy()
        # channels.shape = [n,8]
        flag = 0
        try:
            if int(sys.argv[3]) == 2:
                prot = prepareProteinForAtomtyping(prot)
            channels, prot = getChannels(prot, version=int(sys.argv[3]))
            logger.debug(f"{pdb_path} : get channels with version {sys.argv[3]}")
        except Exception as e:
            logger.debug(f"{pdb_path} : {traceback.format_exc()}, try to get channels f1")
            try:
                flag = 1
                prot = prot_init.copy()
                prot_bool = prot.atomselect('protein') | prot.atomselect(f"element {' '.join(metal_atypes)}")
                prot.filter(prot_bool)
                if int(sys.argv[3]) == 2:
                    prot = prepareProteinForAtomtyping(prot)
                prot_bool = prot.atomselect('protein') | prot.atomselect(f"element {' '.join(metal_atypes)}")
                prot.filter(prot_bool)
                channels, prot = getChannels(prot, version=int(sys.argv[3]))
                logger.debug(f"{pdb_path} : get channels with version {sys.argv[3]}")
            except:
                flag = 2
                logger.debug(f"{pdb_path} : {traceback.format_exc()} : failed to get channels {pdb_path}")
        
        ligand_coords = mol.coords.squeeze()
        receptor_coords = prot.coords.squeeze() #[n,3]
        logger.debug(f"receptor_coords.shape={receptor_coords.shape},ligand_coords.shape={ligand_coords.shape}")
        dist_mat = distance_matrix(ligand_coords,receptor_coords)
        dist_atom_to_ligand = np.min(dist_mat,axis=0)
        
        #if len(sys.argv) < 4 or int(sys.argv[4]) == 0:
        #    pocket_residue_numbers = np.unique(prot.resid[dist_atom_to_ligand < dist])
        #else:
        #    pocket_residue_numbers = np.unique(prot.resid[(dist_atom_to_ligand < dist) & (prot.element!="H")])
        #receptor_pocket_flag = np.array([1 if prot.resid[i] in pocket_residue_numbers else 0 for i in range(prot.numAtoms)]) 
        
        # まず条件に合う原子のインデックスを取得
        if len(sys.argv) < 5 or int(sys.argv[4]) == 0:
            pocket_atom_indices = np.where(dist_atom_to_ligand < dist)[0]
        else:
            pocket_atom_indices = np.where((dist_atom_to_ligand < dist) & (prot.element != "H"))[0]

        # 条件に合う原子のresidとchainの組み合わせを取得
        pocket_resid_chain_pairs = set(
            (prot.resid[i], prot.chain[i]) 
            for i in pocket_atom_indices
        )

        # 全原子に対してフラグを設定
        receptor_pocket_flag = np.array([
            1 if (prot.resid[i], prot.chain[i]) in pocket_resid_chain_pairs 
            else 0 
            for i in range(prot.numAtoms)
        ])
        
        receptor_coords = torch.from_numpy(receptor_coords).float()
        receptor_pocket_flag = torch.from_numpy(receptor_pocket_flag).float()
        receptor_residue_number = torch.from_numpy(prot.resid)
        ligand_coords = torch.from_numpy(ligand_coords).float()
        channels = torch.from_numpy(channels).float()
        #chains = torch.from_numpy(convert_chain_ids_to_integers(prot.chain))
        chains = list(prot.chain)
        logger.debug(f"receptor_coords:{receptor_coords.size()},receptor_pocket_flag:{receptor_pocket_flag.size()},receptor_residue_number:{receptor_residue_number.size()},channels:{channels.size()}")
        return receptor_coords, receptor_pocket_flag, receptor_residue_number, channels, ligand_coords, flag, chains
    except Exception as e:
        logger.debug(f"{traceback.format_exc()}, cannot prepare protein {pdb_path}")
        return None, None, None, None, None, None, None



def featurizer_wrapper(pdb_path):
    try:
        os.makedirs("/".join(pdb_path.split("/")[:-1]).replace("/prospeccts/","/processed/prospeccts/"),exist_ok=True)
        logger.debug("convert pdb file : "+pdb_path)
        coords,pocket_flag,residue_number,channels, ligand_coords, flag, chains = default_featurizer(pdb_path)
        if dist==8:
            save_path = pdb_path.replace("/prospeccts/","/processed/prospeccts/").replace(".pdb",f".pickle")
        else:
            save_path = pdb_path.replace("/prospeccts/","/processed/prospeccts/").replace(".pdb",f"_{dist}.pickle")
        if int(sys.argv[3]) != 2:
            save_path = save_path.replace(".pickle",f"_v{sys.argv[3]}.pickle")
        elif flag != 0:
            save_path = save_path.replace(".pickle",f"_f{flag}.pickle")
        if int(sys.argv[4]) == 1:
            save_path = save_path.replace(".pickle",f"_nH.pickle") # choose pocket residue except not Hydrogen
        if os.path.isfile(save_path):
            os.remove(save_path)
        ligand_save_path = pdb_path.replace("/prospeccts/","/processed/prospeccts/").replace(".pdb","_ligand.pickle")
        if coords is None:
            return False
        torch.save({
            "coords":coords,
            "pocket_flag":pocket_flag,
            "residue_number":residue_number,
            "chains":chains,
            "channels":channels,
        },save_path)
        if not os.path.exists(ligand_save_path):
            torch.save({
                "coords":ligand_coords
            },ligand_save_path)
        return True
    except:
        logger.debug(f"{traceback.format_exc()}, error occured {pdb_path}")
        return False

if __name__ == "__main__":
    try:
        utc_time = datetime.now()
        execution_time_str = (utc_time).strftime('%Y%m%d_%H%M')
        log_folder = f"./logs/prospeccts{execution_time_str}_{sys.argv[3]}"
        os.makedirs(log_folder,exist_ok=True)
        new_path = shutil.copy(f"{sys.argv[0]}",log_folder)
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=log_format, filename=f"{log_folder}/output.log", filemode="w",force=True)
        
        
        Prospeccts_PATH = sys.argv[1] #path to /TOUGH-M1/
        if len(sys.argv) > 2:
            dist = int(sys.argv[2])
        os.makedirs(Prospeccts_PATH.replace("/prospeccts/","/processed/prospeccts/"),exist_ok=True)
        pdb_files = search_pdb_files(Prospeccts_PATH)
        logger.debug(pdb_files)
        logger.info(f"pdb files to be processed : {len(pdb_files)}")
        start = time()
        
        num_cpus = 4
        SUCCESS_COUNT = 0
        with ProcessPoolExecutor(max_workers=4) as executor:
            for result in executor.map(featurizer_wrapper,pdb_files):
                if result:
                    SUCCESS_COUNT += 1
                    logger.info(f"SUCCESS_COUNT={SUCCESS_COUNT}")
            
            
        logger.debug(f"Processed files : {SUCCESS_COUNT}/{len(pdb_files)} {time()-start}s")
    except:
        logger.debug(traceback.format_exc())