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
import string

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, GroupShuffleSplit
from .utils import voc_ap, pdb_check_obsolete, RcsbPdbClusters
from tqdm import tqdm

#import matplotlib.pyplot as plt
import glob
import random
import sys


logger = logging.getLogger(__name__)

class Vertex:
    """
    Vertex dataset by Chen et al
    http://pubs.acs.org/doi/suppl/10.1021/acs.jcim.6b00118/suppl_file/ci6b00118_si_002.zip
    """

    def __init__(self, vertex_data_dir):
        self.vertex_data_dir = vertex_data_dir

    @staticmethod
    def _preprocess_worker(entry):
        def pdb_uniprot_to_chain(pdb_code, uniprot):
            """
            Get pdb chain mapping to uniprot accession using the pdbe api
            """
            result = uniprot
            entity_id = set()
            r = requests.get(f'http://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code}')
            fam = r.json()[pdb_code]['UniProt']

            for fam_id in fam.keys():
                if fam_id != uniprot:
                    continue
                for chain in fam[fam_id]['mappings']:
                    entity_id.add(chain['entity_id'])
            if not entity_id:
                logger.warning(f'No entity_id found for {pdb_code}: {uniprot}')
            return result, entity_id
        
        pdb_code = entry['code'].lower()
        superceded = pdb_check_obsolete(entry['code'])
        if superceded:
            pdb_code = superceded
        result, entity_id = pdb_uniprot_to_chain(pdb_code,entry['uniprot'])
        return [pdb_code, entry['code5'], entry['uniprot'], entity_id]

    def preprocess_once(self):
        """
        Download pdb files and extract pocket around ligands
        """
        logger.info('Preprocessing: downloading data and extracting pockets, this will take time.')
        entries = self.get_structures(moleculekit_features=False,extra_mappings=False)
        
        code5_to_uniprot = {}
        code5_to_seqclust = {}
        uniprot_to_code5 = defaultdict(list)
        clusterer = RcsbPdbClusters(identity=30)  

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            for code, code5, uniprot, entity_id in executor.map(Vertex._preprocess_worker, entries):
                logger.info(f'code5:{code5},uniprot:{uniprot},entity_id:{entity_id}')
                code5_to_uniprot[code5] = uniprot
                uniprot_to_code5[uniprot] = uniprot_to_code5[uniprot] + [code5]
                seqclusts = set([clusterer.get_seqclust(code, c) for c in entity_id])
                if 'None' in seqclusts:
                    seqclusts.remove('None')
                code5_to_seqclust[code5] = seqclusts

        unclustered = [k for k,v in code5_to_seqclust.items() if len(v)==0 ]
        if len(unclustered) > 0:
            logger.info(f"Unable to get clusters for {len(unclustered)} entries: {unclustered}")

        pickle.dump({
                'code5_to_uniprot': code5_to_uniprot,
                'uniprot_to_code5': uniprot_to_code5,
                'code5_to_seqclust': code5_to_seqclust
            },open(os.path.join(self.vertex_data_dir , 'pdbcode_mappings.pickle'), 'wb'))

    def get_structures(self, moleculekit_features=True, extra_mappings=True, moleculekit_version="v2", include_f1=True):
        """
        Get list of PDB structures with metainfo
        """

        root = self.vertex_data_dir

        # Read in a set of (pdb_chain, uniprot, ligand_cc) tuples
        vertex_pdbs = set()
        with open(os.path.join(root, 'protein_pairs.tsv')) as f:
            for i, line in enumerate(f.readlines()):
                if i > 1:
                    tokens = line.split('\t')
                    vertex_pdbs.add((tokens[0].lower(), tokens[2], tokens[1]))
                    vertex_pdbs.add((tokens[5].lower(), tokens[7], tokens[6]))
        logger.debug(f"vertex_pdbs={vertex_pdbs}")
        code5_to_seqclusts = None        
        if extra_mappings:
            mapping = pickle.load(open(os.path.join(self.vertex_data_dir, 'pdbcode_mappings.pickle'), 'rb'))
            code5_to_seqclusts = mapping['code5_to_seqclust']

        # Generate entries for the Vertex set
        entries = []
        for n, (code5, uniprot, ligand_cc) in enumerate(vertex_pdbs):
            pdb_code = code5[:4]
            moleculekit_feature_path = os.path.join(root.replace('/Vertex/','/processed/Vertex/'),f'{pdb_code}/{pdb_code}_site_{int(code5[5])}.pickle')
            ligand_coords_path = os.path.join(root.replace('/Vertex/','/processed/Vertex/'),f'{pdb_code}/{pdb_code}_lig_{int(code5[5])}.pickle')
            if moleculekit_version == "v2" and include_f1 and not os.path.isfile(moleculekit_feature_path):
                moleculekit_feature_path = moleculekit_feature_path.replace(".pickle","_f1.pickle")
            elif moleculekit_version == "v1":
                moleculekit_feature_path = moleculekit_feature_path.replace(".pickle","_v1.pickle")
            logger.debug(f"{moleculekit_feature_path}, {os.path.isfile(moleculekit_feature_path)}")
            if not moleculekit_features or os.path.isfile(moleculekit_feature_path):
                entries.append({
                    'protein': os.path.join(root , f'{pdb_code}/{pdb_code}_clean.pdb'),
                    'pocket': os.path.join(root , f'{pdb_code}/{pdb_code}_site_{int(code5[5])}.pdb'),
                    'ligand': os.path.join(root , f'{pdb_code}/{pdb_code}_lig_{int(code5[5])}.pdb'),
                    'code5': code5,
                    'code': code5[:4],
                    'lig_cc': ligand_cc,
                    'uniprot': uniprot,
                    'seqclusts': code5_to_seqclusts[code5] if code5_to_seqclusts else 'None',
                    'moleculekit_features': moleculekit_feature_path if moleculekit_features else None,
                    'ligand_coords': ligand_coords_path if moleculekit_features else None,
                    'p2rank_prediction': os.path.join(root.replace('/Vertex/',f'/processed/predict_vertex_p2rank_prediction/') , f'{pdb_code}_clean.pdb_residues.csv')
                })
        return entries

    def evaluate_matching(self,descriptor_entries, matcher):
        """
        Evaluate pocket matching on Vertex dataset
        The evaluation metric is AUC

        :param descriptor_entries: List of entries
        :param matcher: PocketMatcher instance
        """

        target_dict = {d['code5']: i for i, d in enumerate(descriptor_entries)}
        prot_pairs = defaultdict(list)
        prot_positives = {}

        # Assemble dictionary pair-of-uniprots -> list_of_pairs_of_indices_into_descriptor_entries
        with open(os.path.join(self.vertex_data_dir, 'protein_pairs.tsv')) as f:
            for i, line in enumerate(f.readlines()):
                if i > 1:
                    tokens = line.split('\t')
                    pdb1, id1, pdb2, id2, cls = tokens[0].lower(), tokens[2], tokens[5].lower(), tokens[7], int(tokens[-1])
                    if pdb1 in target_dict and pdb2 in target_dict:
                        key = (id1, id2) if id1 < id2 else (id2, id1)
                        prot_pairs[key] = prot_pairs[key] + [(target_dict[pdb1], target_dict[pdb2])]
                        if key in prot_positives:
                            assert prot_positives[key] == (cls == 1)
                        else:
                            prot_positives[key] = (cls == 1)

        positives = []
        scores = []
        keys_out = []

        # Evaluate each protein pairs (taking max over all pdb pocket scores, see Fig 1B in Chen et al)
        for key, pdb_pairs in tqdm(prot_pairs.items()):
            unique_idxs = list(set([p[0] for p in pdb_pairs] + [p[1] for p in pdb_pairs]))

            complete_scores = matcher.complete_match([descriptor_entries[i] for i in unique_idxs])

            sel_scores = []
            for pair in pdb_pairs:
                i, j = unique_idxs.index(pair[0]), unique_idxs.index(pair[1])
                if np.isfinite(complete_scores[i, j]):
                    sel_scores.append(complete_scores[i, j])

            if len(sel_scores) > 0:
                positives.append(prot_positives[key])
                keys_out.append(key)
                scores.append(max(sel_scores))
            else:
                logger.warning(f'Skipping a pair, could not be evaluated')

        # Calculate metrics
        fpr, tpr, roc_thresholds = roc_curve(positives, scores)
        auc = roc_auc_score(positives, scores)
        precision, recall, thresholds = precision_recall_curve(positives, scores)
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
            'pairs': keys_out,
            'scores': scores,
            'pos_mask': positives
        }
        return results
    
if __name__ == "__main__":
    Vertex(sys.argv[1]).preprocess_once()