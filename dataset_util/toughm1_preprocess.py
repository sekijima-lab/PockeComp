from toughm1_utils import ToughM1

def create_tough_dataset(db_preprocessing=False, fold_nr=0, n_folds=5, seed=0):

    if db_preprocessing:
        ToughM1().preprocess_once()

    pdb_train, pdb_test = ToughM1().get_structures_splits(fold_nr, strategy='seqclust', n_folds=n_folds, seed=seed)

    print(pdb_train, pdb_test)

create_tough_dataset(db_preprocessing=True)