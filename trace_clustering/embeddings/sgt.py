import logging
import tqdm
import itertools as it
import numpy as np
import pandas as pd
from typing import Set, Tuple, Sequence, Hashable, List, Collection, Dict, Container, Optional
from trace_clustering.util.mp import run_parallel

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

# type definitions
Item = Hashable
Alphabet = Sequence[Item]
Itemset = Set[Item]
ItemSequence = Sequence[Item]
ItemsetSequence = Sequence[Item]
ValidSequence = ItemSequence or ItemsetSequence
Feature = Tuple[Item, Item]


def _get_sequences_alphabet(sequences: Collection[ValidSequence], processes: Optional[int] = -1) -> Alphabet:
    # gets all symbols in the sequences
    alphabets = run_parallel(_get_sequence_alphabet, list(sequences), processes, use_tqdm=False)
    return list(set.union(*alphabets))


def _get_sequence_alphabet(seq: ValidSequence) -> Set:
    # get symbols from given sequence
    if not isinstance(seq[0], Item) and isinstance(seq[0], Container):  # if sequence of itemsets
        return set.union(*map(set, seq))  # add alphabet for each itemset
    else:
        return set(seq)  # otherwise sequence of items


class SGT(object):
    """
    An implementation of the Sequence Graph Transform (SGT) algorithm in [1], extended to support sequences of itemsets,
    where each element in a sequence is actually a set of different symbols. Features are then computed between all
    pairs of symbols composing the alphabet as in the original algorithm.
    [1] - Ranjan, C., Ebrahimi, S., & Paynabar, K. (2016). Sequence graph transform (SGT): A feature extraction
    function for sequence data mining (Extended version). arXiv preprint arXiv:1608.03533.
    """

    alphabet: Alphabet
    features: List[Feature]
    feature_idxs: Dict[Item, int]

    def __init__(self, kappa: float = 1, length_sensitive: bool = True, num_processes: Optional[int] = -1,
                 alphabet: Alphabet = None, verbose: bool = False):
        """
        Initializes the SGT algorithm with the given parameters.
        :param float kappa: tuning parameter for SGT to change the extraction of long-term dependencies. The higher the
        value the lesser the long-term dependency captured in the embedding. Typical values for kappa are 1, 5, 10.
        :param bool length_sensitive: If `True` the embedding produced by SGT will have the information of the sequence
        length. If set to `False` then the embedding of two sequences with similar pattern but different lengths will
        be the same. `False` is similar to length-normalization.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param Alphabet alphabet: Optional, defines the symbols to be used in all SGT embedding computations. If `None`,
        then alphabet will be computed the first time :meth:`trace_clustering.sgt.SGT.fit` is called.
        :param bool verbose: whether to show a progress bar when computing the sequences' embeddings.
        """
        self.kappa = kappa
        self.length_sensitive = length_sensitive
        self.num_processes = num_processes
        self.verbose = verbose
        self.alphabet = alphabet
        if alphabet is not None:
            self._alphabet_updated()

    def _log(self, msg: str):
        if self.verbose:
            logging.info(msg)

    def fit(self, sequences: Collection[ValidSequence], sort: bool = True) -> np.ndarray:
        """
        Computes embeddings using SGT for the given sequences. If not provided during initialization, the alphabet
        is computed directly from all sequences.
        :param Collection[ValidSequence] sequences: the collection of sequences for which to compute embeddings.
        :param bool sort: whether to sort alphabet's symbols before computing pairs.
        :rtype: Tuple[np.ndarray, List[Tuple], List[Hashable]]
        :return: an array shaped (num_sequences, num_features) containing the embeddings resulting for each sequence
        through SGT.
        """

        # if alphabet not provided, get it from sequences
        if self.alphabet is None:
            self._log(f'Computing alphabet from {len(sequences)} sequences...')
            alphabet = _get_sequences_alphabet(sequences, self.num_processes)
            self.alphabet = sorted(alphabet) if sort else alphabet
            self._log(f'Got alphabet of size: {len(self.alphabet)}')
            self._alphabet_updated()

        # get embeddings for each sequence
        self._log('Getting embeddings...')
        embeddings = run_parallel(self.fit_sequence, list(sequences), self.num_processes, use_tqdm=True)
        return np.array(embeddings)

    def fit_sequence(self, sequence: ValidSequence) -> np.ndarray:
        """
        Computes the SGT embedding for the given sequence.
        :param ValidSequence sequence: the sequence for which to compute the SGT embedding.
        :rtype: np.ndarray
        :return: an array shaped (num_features, ) containing the embedding resulting for the given sequence through SGT.
        """
        # get indices for each symbol in sequence
        indices = self._get_sequence_indices(sequence)

        # for each symbol pair in sequence, compute differences and counts of co-occurrences
        symbol_idxs = [i for i in range(len(indices)) if len(indices[i]) > 0]
        embeddings = np.zeros(len(self.features))
        for s1_idx, s2_idx in it.product(symbol_idxs, symbol_idxs):
            feat_idx = self.feature_idxs[(self.alphabet[s1_idx], self.alphabet[s2_idx])]
            s1_idxs = indices[s1_idx]
            s2_idxs = indices[s2_idx]
            diffs = np.array([j - i for j in s2_idxs for i in s1_idxs if j > i])
            if len(diffs) == 0:  # check if pattern never happens in sequence
                continue
            phi = np.sum(np.exp(-self.kappa * diffs))
            psi = phi / (len(diffs) if self.length_sensitive else len(diffs) / len(sequence))
            embeddings[feat_idx] = psi

        return embeddings

    def get_dataframe(self, embeddings: np.ndarray):
        """
        Gets a pandas dataframe from the given embeddings produced by SGT, where columns are each feature (symbol pair)
        and rows contain the embeddings for each sequence.
        :param np.ndarray embeddings: an array shaped (num_sequences, num_features) containing the SGT embeddings.
        :rtype: pd.DataFrame
        :return: a pandas dataframe from the given embeddings.
        """
        return pd.DataFrame(embeddings, columns=self.features)

    def get_dataframes(self, embeddings: np.ndarray):
        """
        Gets a pandas dataframe for each given embedding produced by SGT, where columns and rows correspond to each
        symbol of the alphabet, and the values indicate the embedding value for the corresponding (row, column) feature.
        :param np.ndarray embeddings: an array shaped (num_sequences, num_features) containing the SGT embeddings.
        :rtype: List[pd.DataFrame]
        :return: a pandas dataframe from each given embedding.
        """
        alphabet_len = len(self.alphabet)
        dfs = []
        for i, embedding in enumerate(embeddings):
            dfs.append(pd.DataFrame(embedding.reshape((alphabet_len, alphabet_len)),
                                    columns=self.alphabet, index=self.alphabet))
        return dfs

    def _alphabet_updated(self):
        self.features = list(it.product(self.alphabet, self.alphabet))
        self.feature_idxs = {feat: idx for idx, feat in enumerate(self.features)}

    def _get_sequence_indices(self, sequence: ValidSequence) -> List[np.ndarray]:
        indices = []
        for symbol in self.alphabet:
            if not isinstance(sequence[0], Item) and isinstance(sequence[0], Container):
                # if sequence of itemsets, get indexes one by one
                indices.append(np.array([i for i in range(len(sequence)) if symbol in sequence[i]]))
            else:
                indices.append(np.where(np.asarray(sequence) == symbol)[0])  # use numpy
        return indices
