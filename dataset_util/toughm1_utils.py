# adopted from DeeplyTough repository

import concurrent.futures
import logging
import os
import pickle
import subprocess
import tempfile
import urllib.request
from collections import defaultdict

import Bio.PDB as PDB
import numpy as np
import requests

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, GroupShuffleSplit

#import matplotlib.pyplot as plt
import glob
import random
import traceback
from functools import reduce

from .utils import RcsbPdbClusters, pdb_check_obsolete, voc_ap, random_eliminate

logger = logging.getLogger(__name__)

class ToughM1:
    """
    TOUGH-M1 dataset by Govindaraj and Brylinski
    https://osf.io/6ngbs/wiki/home/
    """
    def __init__(self, tough_data_dir="./TOUGH-M1"):
        self.tough_data_dir = tough_data_dir

    @staticmethod
    def _preprocess_worker(entry):

        def struct_to_centroid(structure):
            return np.mean(np.array([atom.get_coord() for atom in structure.get_atoms()]), axis=0)

        def pdb_chain_to_uniprot(pdb_code, query_chain_id):
            """
            Get pdb chain mapping to uniprot accession using the pdbe api
            """
            result = 'None'
            entity_id = 'None'
            r = requests.get(f'http://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code}')
            fam = r.json()[pdb_code]['UniProt']

            for fam_id in fam.keys():
                for chain in fam[fam_id]['mappings']:
                    if chain['chain_id'] == query_chain_id:
                        if result != 'None' and fam_id != result:
                            logger.warning(f'DUPLICATE {fam_id} {result}')
                        result = fam_id
                        entity_id = chain['entity_id']
            if result == 'None':
                logger.warning(f'No uniprot accession found for {pdb_code}: {query_chain_id}')
            return result,entity_id

        # 1) We won't be using provided `.fpocket` files because they don't contain the actual atoms, just
        # Voronoii centers. So we run fpocket2 ourselves, it seems to be equivalent to published results.
        #try:
        #    command = ['fpocket2', '-f', entry['protein']]
        #    subprocess.run(command, check=True)
        #except subprocess.CalledProcessError as e:
        #    logger.warning('Calling fpocket2 failed, please make sure it is on the PATH')
        #    raise e

        # 2) Some chains have been renamed since TOUGH-M1 dataset was released so one cannot directly retrieve
        # uniprot accessions corresponding to a given chain. So we first locate corresponding chains in the
        # original pdb files, get their ids and translate those to uniprot using the SIFTS webservices.
        parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
        tough_str = parser.get_structure('t', entry['protein'])
        tough_c = struct_to_centroid(tough_str)

        # 2a) Some structures are now obsolete since TOUGH-M1 was published, for these, get superceding entry
        pdb_code = entry['code'].lower()
        superceded = pdb_check_obsolete(entry['code'])
        if superceded:
            pdb_code = superceded
        # 2b) try to download pdb from RSCB mirror site
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = tmpdir + '/prot.pdb'
            try:
                urllib.request.urlretrieve(f"http://files.rcsb.org/download/{pdb_code}.pdb", fname)
            except:
                logger.info(f'Could not download PDB: {pdb_code}')
                return [entry['code5'], 'None', 'None','None']
            orig_str = parser.get_structure('o', fname)

        # TOUGH authors haven't re-centered the chains so we can roughly find them just by centroids :)
        dists = []
        ids = []
        for model in orig_str:
            for chain in model:
                if len(chain) < 20:  # ignore chains with fewer than 20 residues
                    continue
                dists.append(np.linalg.norm(struct_to_centroid(chain) - tough_c))
                ids.append(chain.id)
        chain_id = ids[np.argmin(dists)]
        if np.min(dists) > 5:
            logger.warning(f"Suspiciously large distance when trying to map tough structure to downloaded one"
                           f"DIST {dists} {ids} {entry['code']} {pdb_code}")
            return [entry['code5'], 'None', 'None','None']
        try:
            uniprot, entity_id = pdb_chain_to_uniprot(pdb_code.lower(), chain_id)
        except Exception as e:
            logger.info(f"{traceback.format_exc()}, cannot find either uniprot and entity_id, {entry['code5']} {pdb_code.lower() + chain_id}")
            uniprot, entity_id = 'None', 'None'
        return [entry['code5'], uniprot, pdb_code.lower() + chain_id, entity_id]

    def get_pdbcode_mappings(self):
        code5_to_uniprot = {}
        code5_to_seqclust = {}
        uniprot_to_code5 = defaultdict(list)
        logger.info('Preprocessing: obtaining uniprot accessions, this will take time.')
        entries = self.get_structures(extra_mappings=False, moleculekit_features=False)
        clusterer = RcsbPdbClusters(identity=30)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for code5, uniprot, code5new,entity_id in executor.map(ToughM1._preprocess_worker, entries):
                logger.info(f'code5:{code5},uniprot:{uniprot},code5new:{code5new},entity_id:{entity_id}')
                code5_to_uniprot[code5] = uniprot
                uniprot_to_code5[uniprot] = uniprot_to_code5[uniprot] + [code5]
                code5_to_seqclust[code5] = clusterer.get_seqclust(code5new[:4], entity_id)


        unclustered = [k for k,v in code5_to_seqclust.items() if v == 'None']
        if len(unclustered) > 0:
            logger.info(f"Unable to get clusters for {len(unclustered)} entries: {unclustered}")

        # write uniprot mapping to file
        pickle.dump({
                'code5_to_uniprot': code5_to_uniprot,
                'uniprot_to_code5': uniprot_to_code5,
                'code5_to_seqclust': code5_to_seqclust
            },
            open(os.path.join(self.tough_data_dir, 'pdbcode_mappings.pickle'), 'wb')
        )

    def preprocess_once(self, mapping_exist_skip = True):
        """
        Re-run fpocket2 and try to obtain Uniprot Accession for each PDB entry.
        Needs to be called just once in a lifetime
        """
        if not mapping_exist_skip or not os.path.exists(os.path.join(self.tough_data_dir, 'pdbcode_mappings.pickle')):
            self.get_pdbcode_mappings()

    def get_structures(self, extra_mappings=True, moleculekit_features=True, valid_data=None, moleculekit_version="v2", include_f1=False):
        """
        Get list of PDB structures with metainfo
        """
        root = os.path.join(self.tough_data_dir, 'TOUGH-M1_dataset')
        fname_uniprot_mapping = os.path.join(self.tough_data_dir, 'pdbcode_mappings.pickle')

        # try to load translation pickle
        code5_to_uniprot = None
        code5_to_seqclust = None
        if extra_mappings:
            mapping = pickle.load(open(fname_uniprot_mapping, 'rb'))
            code5_to_uniprot = mapping['code5_to_uniprot']
            code5_to_seqclust = mapping['code5_to_seqclust']

        entries = []    
        # こんな感じで特徴量増える度に追加していく 
        with open(os.path.join(self.tough_data_dir, 'TOUGH-M1_pocket.list')) as f:
            for line in f.readlines():
                code5, pocketnr, _ = line.split()
                moleculekit_feature_path = root.replace('/TOUGH-M1/','/processed/TOUGH-M1/') + f'/{code5}/{code5}.pickle'
                ligand_coords_path = root.replace('/TOUGH-M1/','/processed/TOUGH-M1/') + f'/{code5}/{code5}_ligand.pickle'
                if moleculekit_version == "v2" and include_f1 and not os.path.isfile(moleculekit_feature_path):
                    moleculekit_feature_path = moleculekit_feature_path.replace(".pickle","_f1.pickle")
                elif moleculekit_version == "v1":
                    moleculekit_feature_path = moleculekit_feature_path.replace(".pickle","_v1.pickle")
                if not moleculekit_features or os.path.isfile(moleculekit_feature_path):
                    entries.append({
                        'protein': root + f'/{code5}/{code5}.pdb',
                        'ligand': root + f'/{code5}/{code5}00.pdb',
                        'code5':code5,
                        'code':code5[:4],
                        'uniprot': code5_to_uniprot[code5] if code5_to_uniprot else 'None',
                        'seqclust': code5_to_seqclust[code5] if code5_to_seqclust else 'None',
                        'moleculekit_features': moleculekit_feature_path if moleculekit_features else None,
                        'ligand_coords': ligand_coords_path if moleculekit_features else None,
                        'p2rank_prediction': os.path.join(root.replace('/TOUGH-M1/TOUGH-M1_dataset','/processed/predict_toughm1_p2rank_prediction') ,f'{code5}.pdb_residues.csv')
                    })
            
                    
        if valid_data=="vertex":
            fname_uniprot_mapping = os.path.join(self.tough_data_dir.replace("/TOUGH-M1/","/Vertex/"), 'pdbcode_mappings.pickle')
            mapping = pickle.load(open(fname_uniprot_mapping, 'rb'))
            entries = [entry for entry in entries if entry["seqclust"] not in reduce(set.union,list(mapping["code5_to_seqclust"].values()))]
        elif valid_data=="prospeccts":
            fname_uniprot_mapping = os.path.join(self.tough_data_dir.replace("/TOUGH-M1/","/prospeccts/"), 'pdbcode_mappings.pickle')
            mapping = pickle.load(open(fname_uniprot_mapping, 'rb'))
            entries = [entry for entry in entries if entry["seqclust"] not in reduce(set.union,list(mapping["code5_to_seqclust"].values()))]
        elif valid_data=="scPDB":
            fname_uniprot_mapping = os.path.join(self.tough_data_dir.replace("/TOUGH-M1/","/segmentation_data_dir/scPDB/"), 'pdbcode_mappings.pickle')
            mapping = pickle.load(open(fname_uniprot_mapping, 'rb'))
            entries = [entry for entry in entries if entry["seqclust"] not in reduce(set.union,list(mapping["code5_to_seqclust"].values()))]

        return entries
    
    # segmentation for valid_data用
    def get_structures_eliminate(self, strategy='seqclust', seed=0,
                       moleculekit_features=True, valid_data=None, moleculekit_version="v1",include_f1=False):
        pdb_entries = self.get_structures(extra_mappings=True, 
                                          moleculekit_features=moleculekit_features, 
                                          valid_data=valid_data, 
                                          moleculekit_version=moleculekit_version, 
                                          include_f1=include_f1)
        test_entries = [entry for entry in self.get_structures(extra_mappings=True, 
                                          moleculekit_features=moleculekit_features, 
                                          valid_data=None, 
                                          moleculekit_version=moleculekit_version, 
                                          include_f1=include_f1) if entry not in pdb_entries]
        if strategy == 'pdb_folds':
            train_entries = pdb_entries
        elif strategy == 'uniprot_folds':
            train_entries = random_eliminate(pdb_entries, "uniprot", seed)
        elif strategy == 'seqclust':
            train_entries = random_eliminate(pdb_entries, "seqclust", seed)
        elif strategy == 'none':
            train_entries = pdb_entries
        else:
            raise NotImplementedError
        return train_entries, test_entries

    def get_structures_splits(self, fold_nr, valid_data=None, strategy='seqclust', n_folds=5, seed=0, moleculekit_version="v2", include_f1=False):
        pdb_entries = self.get_structures(valid_data=valid_data, moleculekit_version=moleculekit_version, include_f1=include_f1)

        if strategy == 'pdb_folds':
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            folds = list(splitter.split(pdb_entries))
            train_idx, test_idx = folds[fold_nr]
            return [pdb_entries[i] for i in train_idx], [pdb_entries[i] for i in test_idx]

        elif strategy == 'uniprot_folds':
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=1.0/n_folds, random_state=seed)
            pdb_entries = list(filter(lambda entry: entry['uniprot'] != 'None', pdb_entries))
            folds = list(splitter.split(pdb_entries, groups=[e['uniprot'] for e in pdb_entries]))
            train_idx, test_idx = folds[fold_nr]
            return [pdb_entries[i] for i in train_idx], [pdb_entries[i] for i in test_idx]

        elif strategy == 'seqclust':
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=1.0/n_folds, random_state=seed)
            pdb_entries = list(filter(lambda entry: entry['seqclust'] != 'None', pdb_entries))
            folds = list(splitter.split(pdb_entries, groups=[e['seqclust'] for e in pdb_entries]))
            train_idx, test_idx = folds[fold_nr]
            return [pdb_entries[i] for i in train_idx], [pdb_entries[i] for i in test_idx]

        elif strategy == 'none':
            return pdb_entries, pdb_entries
        else:
            raise NotImplementedError
        
    # splitした後にeliminate
    def get_structures_splits_eliminate(self, fold_nr, valid_data=None, strategy='seqclust', n_folds=5, seed=0, moleculekit_version="v1", include_f1=False):
        
        pdb_entries = self.get_structures(valid_data=valid_data, moleculekit_version=moleculekit_version, include_f1=include_f1)

        if strategy == 'pdb_folds':
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            folds = list(splitter.split(pdb_entries))
            train_idx, test_idx = folds[fold_nr]
            return [pdb_entries[i] for i in train_idx], [pdb_entries[i] for i in test_idx]

        elif strategy == 'uniprot_folds':
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=1.0/n_folds, random_state=seed)
            pdb_entries = list(filter(lambda entry: entry['uniprot'] != 'None', pdb_entries))
            #pdb_entries = random_eliminate(pdb_entries, 'uniprot')
            folds = list(splitter.split(pdb_entries, groups=[e['uniprot'] for e in pdb_entries]))
            train_idx, test_idx = folds[fold_nr]
            train_entries = random_eliminate([pdb_entries[i] for i in train_idx],'uniprot')
            return train_entries, [pdb_entries[i] for i in test_idx]

        elif strategy == 'seqclust':
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=1.0/n_folds, random_state=seed)
            pdb_entries = list(filter(lambda entry: entry['seqclust'] != 'None', pdb_entries))
            #pdb_entries = random_eliminate(pdb_entries, 'seqclust')
            folds = list(splitter.split(pdb_entries, groups=[e['seqclust'] for e in pdb_entries]))
            train_idx, test_idx = folds[fold_nr]
            train_entries = random_eliminate([pdb_entries[i] for i in train_idx],'seqclust')
            return train_entries, [pdb_entries[i] for i in test_idx]
        elif strategy == 'none':
            return pdb_entries, pdb_entries
        else:
            raise NotImplementedError

    def evaluate_matching(self, descriptor_entries, matcher):
        """
        Evaluate pocket matching on TOUGH-M1 dataset. The evaluation metrics is AUC.

        :param descriptor_entries: List of entries
        :param matcher: PocketMatcher instance
        """

        target_dict = {d['code5']: d for d in descriptor_entries}
        pairs = []
        positives = []

        def parse_file_list(f):
            f_pairs = []
            for line in f.readlines():
                id1, id2 = line.split()[:2]
                if id1 in target_dict and id2 in target_dict:
                    f_pairs.append((target_dict[id1], target_dict[id2]))
            return f_pairs

        with open(os.path.join(self.tough_data_dir, 'TOUGH-M1_positive.list')) as f:
            pos_pairs = parse_file_list(f)
            pairs.extend(pos_pairs)
            positives.extend([True] * len(pos_pairs))

        with open(os.path.join(self.tough_data_dir, 'TOUGH-M1_negative.list')) as f:
            neg_pairs = parse_file_list(f)
            pairs.extend(neg_pairs)
            positives.extend([False] * len(neg_pairs))
        logger.info(f'positives shape : {np.shape(positives)}')
        scores = matcher.pair_match(pairs)
        for s in scores:
            logger.info(f'{s}->{np.isfinite(s)}')
        goodidx = np.flatnonzero(np.isfinite(np.squeeze(np.array(scores))))
        if len(goodidx) != len(scores):
            logger.warning(f'Ignoring {len(scores) - len(goodidx)} pairs')
            positives_clean, scores_clean = np.array(positives)[goodidx],  np.array(scores)[goodidx]
        else:
            positives_clean, scores_clean = positives, scores

        # Calculate metrics
        print(f'positives_clean:{positives_clean},scores_clean:{scores_clean}')


        fpr, tpr, roc_thresholds = roc_curve(positives_clean, scores_clean)

        """
        # Plotting ROC curve 
        print(f'fpr:{fpr}')
        print(f'tpr:{tpr}')
        print(f'roc_thresholds:{roc_thresholds}')
        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()
        plt.savefig('sklearn_roc_curve_toughm1.png')
        """
        
        
        auc = roc_auc_score(positives_clean, scores_clean) #sklearnのパッケージ
        precision, recall, thresholds = precision_recall_curve(positives_clean, scores_clean)
        ap = voc_ap(recall[::-1], precision[::-1])

        results = {
            'ap': ap,
            'pr': precision,
            're': recall,
            'th': thresholds,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'th_roc': roc_thresholds,
            'pairs': pairs,
            'scores': scores,
            'pos_mask': positives
        }
