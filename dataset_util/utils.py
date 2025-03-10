import numpy as np
import requests
import logging
import os
import urllib
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


def voc_ap(rec, prec):
    """
    Compute VOC AP given precision and recall.
    Taken from https://github.com/marvis/pytorch-yolo2/blob/master/scripts/voc_eval.py
    Different from scikit's average_precision_score (https://github.com/scikit-learn/scikit-learn/issues/4577)
    """
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def pdb_check_obsolete(pdb_code):
    """ Check the status of a pdb, if it is obsolete return the superceding PDB ID else return None """
    try:
        r = requests.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/status/{pdb_code}').json()
    except:
        logger.info(f"Could not check obsolete status of {pdb_code}")
        return None
    if r[pdb_code][0]['status_code'] == 'OBS':
        pdb_code = r[pdb_code][0]['superceded_by'][0]
        logger.info(f'pdb_code returned from pdb_check_obsolete : {pdb_code}')
        return pdb_code
    else:
        return None

class RcsbPdbClusters:
    def __init__(self, identity=30):
        self.identity = identity
        self.clusters = {}
        self._fetch_cluster_file()

    def _download_cluster_sets(self, cluster_file_path):
        os.makedirs(os.path.dirname(cluster_file_path), exist_ok=True)
        # Note that the files changes frequently as do the ordering of cluster within
        #print(f'_download_cluster_sets URL::https://cdn.rcsb.org/resources/sequence/clusters/bc-{self.identity}.out')
        #request.urlretrieve(f'https://cdn.rcsb.org/resources/sequence/clusters/bc-{self.identity}.out', cluster_file_path)
        urllib.request.urlretrieve(f'https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{self.identity}.txt', cluster_file_path)
                              #https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt

    def _fetch_cluster_file(self):
        """ load cluster file if found else download and load """
        #cluster_file_path = os.path.join(self.cluster_dir, f"bc-{self.identity}.out")
        cluster_file_path = os.path.join(os.environ["DATASET_ROOT"], f"clusters-by-entity-{self.identity}.txt")
        logger.info(f"cluster file path: {cluster_file_path}")
        if not os.path.exists(cluster_file_path):
            logger.warning("Cluster definition not found, will download a fresh one.")
            logger.warning("However, this will very likely lead to silent incompatibilities with any old 'pdbcode_mappings.pickle' files! Please better remove those manually.")
            self._download_cluster_sets(cluster_file_path)

        for n, line in enumerate(open(cluster_file_path, 'rb')):
            for id in line.decode('ascii').split():
                self.clusters[id] = n

    def get_seqclust(self, pdbCode, entity_id, check_obsolete=True):
        """ Get sequence cluster ID for a pdbcode chain using RCSB mmseq2/blastclust predefined clusters """
        logger.info(f'pdbCode:{pdbCode}, entity_id:{entity_id}')
        query_str = "{}_{}".format(pdbCode.upper(),entity_id)  # e.g. 1ATP_I
        #query_str = f"{pdbCode.upper()}_{chainId.upper()}" 
        seqclust = self.clusters.get(query_str, 'None')
        
        if check_obsolete and seqclust == 'None':
            superceded = pdb_check_obsolete(pdbCode)
            if superceded is not None:
                logger.info(f"Assigning cluster for obsolete entry via superceding: {pdbCode}->{superceded} {entity_id}")
                return self.get_seqclust(superceded, entity_id, check_obsolete=False)  # assumes chain is same in superceding entry
        if seqclust == 'None':
            logger.info(f"unable to assign cluster to {pdbCode} {entity_id}")
        return seqclust


def random_eliminate(entries,key,seed=0):
    d = defaultdict(lambda:True)
    new_entries = []
    random.seed(seed)
    random.shuffle(entries)
    for entry in entries:
        if type(entry[key])==set:
            flag = True
            for v in entry[key]:
                if d[v] == False:
                    flag = False
                    break
            if flag:
                for v in entry[key]:
                    d[v] = False
                new_entries.append(entry)
                          
        elif d[entry[key]]:
            d[entry[key]] = False
            new_entries.append(entry)
    return entries

def download_pdb(pdb_code, output_dir):
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        output_path = os.path.join(output_dir, f"{pdb_code}_rcsb.pdb")
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return output_path
    else:
        raise Exception(f"Failed to download PDB file for {pdb_code}")

def extract_chains(input_pdb, chain_ids, output_pdb):
    parser = PDB.PDBParser()
    io = PDB.PDBIO()
    structure = parser.get_structure("protein", input_pdb)
    
    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain.id in chain_ids

    io.set_structure(structure)
    io.save(output_pdb, ChainSelect())