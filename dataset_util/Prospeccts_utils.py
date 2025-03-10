import os
import glob
import pickle
import requests
import string
import concurrent.futures
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from .utils import voc_ap, pdb_check_obsolete, RcsbPdbClusters
from collections import defaultdict
import logging
import sys
import Bio.PDB as PDB
import tempfile
import urllib

logger = logging.getLogger(__name__)


class Prospeccts:
    """ ProSPECCTs dataset by Ehrt et al (http://www.ccb.tu-dortmund.de/ag-koch/prospeccts/) """

    dbnames = ['P1', 'P1.2', 'P2', 'P3', 'P4', 'P5', 'P5.2', 'P6', 'P6.2', 'P7']

    def __init__(self, prospeccts_data_dir, dbname='P1'):
        self.prospeccts_data_dir = prospeccts_data_dir
        self.dbname = dbname

    @staticmethod
    def _get_pdb_code_from_raw_pdb(pdbpath):
        search_string = os.path.basename(pdbpath)[:2]
        """
        dict constracted from https://doi.org/10.1371/journal.pcbi.1006483.s011
        """
        """
        logger.info(f'searching for pdb id using string: {search_string}')
        with open(pdbpath, 'r') as f:
            g = f.readlines()
            pdb_code = None
            while pdb_code is None and len(g):
                line = g.pop(0)
                for s in line.split():
                    if search_string in s:
                        maybe_code = s[:4]
                        # check this is a real NMR pdb code
                        try:
                            logger.info(f"checking whether {maybe_code} is a real NMR entry in the PDB")
                            r = requests.get(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/experiment/{maybe_code}")
                            exp = r.json()[maybe_code][0]['experimental_method']
                        except Exception as e:
                            continue
                        # if pdb is real, and the experimental method is NMR. Eureka!
                        if "NMR" in exp:
                            pdb_code = maybe_code
        """
        with open(pdbpath, 'r') as f:
            for line in f:
                if line.startswith("HEADER"):
                    pdb_code = line[62:66]
                    break
        
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdbpath.replace(".pdb","_clean.pdb"))
        for chain in structure.get_chains():
            chain_id = chain.id
            break    
        return pdb_code, chain_id

    @staticmethod
    def _extract_pocket_and_get_uniprot(pdbpath):

        def struct_to_centroid(structure):
            return np.mean(np.array([atom.get_coord() for atom in structure.get_atoms()]), axis=0)

        fname = os.path.basename(pdbpath).split('.')[0]
        if '_' in fname:
            return None, None

        # 1) Extract the pocket
        #detector = PocketFromLigandDetector(include_het_resname=False, save_clean_structure=True,
        #                                    keep_other_hets=False, min_lig_atoms=1, allowed_lig_names=['LIG'])
        #detector.run_one(pdbpath, os.path.dirname(pdbpath))

        # 2) Attempt to map to Uniprots (fails from time to time, return 'None' in that case)
        pdb_code = fname[:4].lower()
        query_chain_id = fname[4].upper() if len(fname) > 4 else ''
        code5 = pdb_code + query_chain_id
        result = set()
        entity_id = set()


        try:
            r = requests.get(f'http://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code}')
            fam = r.json()[pdb_code]['UniProt']
        except Exception as e:
            # this logically fails for artificial proteins not in PDB, such as in decoys (P3, P4), but that's fine.
            logger.warning(f'PDB not found {e} {pdb_code} {query_chain_id}')
            # 2b) In the case of NMR structures, Prospeccts has incomplete PDB IDs (e.g. 'cz00A' is really '1cz2 00 A')
            # Therefore for this dataset, try to get the full PDB ID from the raw PDB text
            if "NMR_structures" in pdbpath or "decoy" in pdbpath:
                pdb_code_new, chain_id = Prospeccts._get_pdb_code_from_raw_pdb(pdbpath)
                logger.info(f"Original PDB ID of {code5} is {pdb_code_new,chain_id}, not {pdb_code,query_chain_id}")
                pdb_code = pdb_code_new
                if not pdb_code:
                    pdb_code = 'XXXX'
                if not query_chain_id:
                    query_chain_id = chain_id
                try:
                    r = requests.get(f'http://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code}')
                    fam = r.json()[pdb_code]['UniProt']
                except Exception as e:
                    logger.warning(f'PDB not found {e} {pdb_code} {query_chain_id}')
                    return pdb_code, code5, result, entity_id
        

        for fam_id in fam.keys():
            for chain in fam[fam_id]['mappings']:
                if not query_chain_id:
                    result.add(fam_id)
                    entity_id.add(chain["entity_id"])
                elif chain['chain_id'] == query_chain_id:
                    if len(result) > 0 and fam_id != next(iter(result)):
                        logger.warning(f'Duplicate chain {fam_id} {result}')
                    result.add(fam_id)
                    entity_id.add(chain["entity_id"])
        if not result:
            logger.warning(f'Chain not found {pdb_code} chain {query_chain_id}')
        return pdb_code, code5, result, entity_id

    def preprocess_once(self):
        if self.dbname == 'P1':  # batch downloading and mapping together and do it just once, e.g. with P1
            logger.info('Preprocessing: extracting pockets and obtaining uniprots, this will take time.')
            all_pdbs = glob.glob(self.prospeccts_data_dir + '/**/**/*.pdb', recursive=True)
            all_pdbs = [pdb for pdb in all_pdbs if (pdb.count('_site') + pdb.count('_lig') + pdb.count('_clean')) == 0]

            clusterer = RcsbPdbClusters(identity=30)   

            code5_to_uniprot = {}
            code5_to_seqclust = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                for code, code5, uniprot, entity_id in executor.map(Prospeccts._extract_pocket_and_get_uniprot, all_pdbs):
                    logger.info(f'code:{code},code5:{code5},uniprot:{uniprot},entity_id:{entity_id}')
                    code5_to_uniprot[code5] = uniprot
                    seqclusts = set([clusterer.get_seqclust(code, c) for c in entity_id])
                    if 'None' in seqclusts:
                        seqclusts.remove('None')
                    code5_to_seqclust[code5] = seqclusts

            unclustered = [k for k,v in code5_to_seqclust.items() if len(v)==0]
            if len(unclustered) > 0:
                logger.info(f"Unable to get clusters for {len(unclustered)} entries: {unclustered}")

            pickle.dump({
                    'code5_to_uniprot': code5_to_uniprot,
                    'code5_to_seqclust': code5_to_seqclust
                },
                open(os.path.join(self.prospeccts_data_dir, 'pdbcode_mappings.pickle'), 'wb')
            )

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
        return dir1, dir2, listfn

    def get_structures(self, moleculekit_features=True, extra_mappings=True, moleculekit_version="v2", include_f1=True):
        """ Get list of PDB structures with metainfo """
        dir1, dir2, listfn = self._prospeccts_paths()
        root = os.path.join(self.prospeccts_data_dir, dir1)

        db_pdbs = set()
        with open(os.path.join(root, listfn)) as f:
            for line in f.readlines():
                tokens = line.split(',')
                db_pdbs.add(tokens[0])
                db_pdbs.add(tokens[1])
                
        code5_to_seqclusts, code5_to_uniprot = None, None
        if extra_mappings:
            mapping = pickle.load(open(os.path.join(self.prospeccts_data_dir, 'pdbcode_mappings.pickle'), 'rb'))
            code5_to_seqclusts = mapping['code5_to_seqclust']
            code5_to_uniprot = mapping['code5_to_uniprot']

        entries = []
        for pdb in db_pdbs:
            moleculekit_feature_path = root.replace('/prospeccts/','/processed/prospeccts/') + f'/{dir2}/{pdb}_clean.pickle'
            ligand_coords_path = root.replace('/prospeccts/','/processed/prospeccts/') + f'/{dir2}/{pdb}_ligand.pickle'
            if moleculekit_version == "v2" and include_f1 and not os.path.isfile(moleculekit_feature_path):
                moleculekit_feature_path = moleculekit_feature_path.replace(".pickle","_f1.pickle")
            elif moleculekit_version == "v1":
                moleculekit_feature_path = moleculekit_feature_path.replace(".pickle","_v1.pickle")
            if not moleculekit_features or os.path.isfile(moleculekit_feature_path):
                entries.append({
                    'protein': root + f'/{dir2}/{pdb}_clean.pdb',
                    'pocket': root + f'/{dir2}/{pdb}_site_1.pdb',
                    'ligand': root + f'/{dir2}/{pdb}_lig_1.pdb',
                    'code5': pdb,
                    'code': pdb[:4],
                    'uniprot': code5_to_uniprot[pdb[:4].lower()+pdb[4:]] if code5_to_uniprot else 'None',
                    'seqclusts': code5_to_seqclusts[pdb[:4].lower()+pdb[4:]] if code5_to_seqclusts else 'None',
                    'moleculekit_features': moleculekit_feature_path if moleculekit_features else None,
                    'ligand_coords': ligand_coords_path if moleculekit_features else None,
                    'p2rank_prediction': self.prospeccts_data_dir.replace('/prospeccts/',f'/processed/predict_{self.dbname}_p2rank_prediction/') + f'{pdb}_clean.pdb_residues.csv'
                })
        return entries

    def evaluate_matching(self, descriptor_entries, matcher):
        """
        Evaluate pocket matching on one Prospeccts dataset
        The evaluation metrics is AUC

        :param descriptor_entries: List of entries
        :param matcher: PocketMatcher instance
        """

        target_dict = {d['code5']: d for d in descriptor_entries}
        pairs = []
        positives = []

        dir1, dir2, listfn = self._prospeccts_paths()
        root = os.path.join(self.prospeccts_data_dir, dir1)

        with open(os.path.join(root, listfn)) as f:
            for line in f.readlines():
                tokens = line.split(',')
                id1, id2, cls = tokens[0], tokens[1], tokens[2].strip()
                if id1 in target_dict and id2 in target_dict:
                    pairs.append((target_dict[id1], target_dict[id2]))
                    positives.append(cls == 'active')
                else:
                    logger.warning(f'Detection entry missing for {id1},{id2}')

        scores = matcher.pair_match(pairs)

        goodidx = np.flatnonzero(np.isfinite(np.array(scores)))
        if len(goodidx) != len(scores):
            logger.warning(f'Ignoring {len(scores) - len(goodidx)} pairs')
            positives_clean, scores_clean = np.array(positives)[goodidx], np.array(scores)[goodidx]
        else:
            positives_clean, scores_clean = positives, scores

        # Calculate metrics
        fpr, tpr, roc_thresholds = roc_curve(positives_clean, scores_clean)
        auc = roc_auc_score(positives_clean, scores_clean)
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
        return results
    
if __name__ == "__main__":
    Prospeccts(sys.argv[1]).preprocess_once()
