import logging
import os
import tqdm
import numpy as np
import pandas as pd
from typing import Tuple, List, Set, Optional
from trace_clustering.bin.split_data import TRACE_IDX_COL
from trace_clustering.embeddings import EmbeddingAlgorithm, SGT_FEAT_PAIR_SEP_STR, SGT_FEAT_VAL_SEP_STR
from trace_clustering.embeddings.sgt import SGT

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class SGTFeatureSetEmbedding(EmbeddingAlgorithm):
    """
    An algorithm that computes a trace embedding using SGT that finds temporal relationships between feature values
    *between* all pairs of high-level feature. This is done by considering each element in a trace sequence as a
    feature set (itemset sequence) rather than considering only individual features (item sequence).
    """

    def __init__(self, output_dir: str,
                 features_filter: Optional[Set[str]],
                 kappa: float = 1,
                 length_sensitive: bool = True,
                 num_processes: int = None,
                 filter_constant: bool = True,
                 max_timesteps: int = -1):
        """
        Initializes the embedding extraction algorithm.
        :param str output_dir: the output in which to save algorithm's results.
        :param set[str] features_filter: the names of the features to be used for clustering.
        :param float kappa: tuning parameter for SGT to change the extraction of long-term dependencies. The higher the
        value the lesser the long-term dependency captured in the embedding. Typical values for kappa are 1, 5, 10.
        :param bool length_sensitive: If `True` the embedding produced by SGT will have the information of the sequence
        length. If set to `False` then the embedding of two sequences with similar pattern but different lengths will
        be the same. `False` is similar to length-normalization.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param bool filter_constant: whether to remove features whose value is constant across all traces in the dataset.
        """
        super().__init__(output_dir, features_filter, num_processes, filter_constant, max_timesteps)
        self.kappa = kappa
        self.length_sensitive = length_sensitive

    def get_embeddings(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get embeddings from the given dataset of traces.
        :param pd.DataFrame traces_df: the pandas dataframe containing the high-level features for all traces/episodes.
        :rtype: Tuple[np.ndarray, List[str]]
        :return: a pandas dataframe containing the features (columns) for each trace.
        """
        # filter features
        features, traces_df = self._pre_process_traces(traces_df)
        traces_df[features] = traces_df[features].astype(str)  # ensure categorical features
        num_features = len(features)

        logging.info('========================================')
        logging.info(f'Adding feature prefix to the values of all {num_features} features...')
        for feature in tqdm.tqdm(features):
            traces_df[feature] = f'{feature}{SGT_FEAT_VAL_SEP_STR}' + traces_df[feature].astype(str)

        logging.info('========================================')
        logging.info('Collecting traces...')
        traces = [(idx, [set(t) for t in df[features].values.tolist()])
                  for idx, df in tqdm.tqdm(traces_df.groupby(TRACE_IDX_COL))]

        # sanity check, see if episodes align with traces indexes
        if not all(idx == t[0] for idx, t in enumerate(traces)):
            traces = sorted(traces, key=lambda x: x[0])

        num_traces = len(traces)
        trace_lens = [len(t) for idx, t in traces]
        logging.info(f'Extracted traces for {num_features} features from the dataset'
                     f'\n\tTotal {num_traces} traces (episodes) per feature'
                     f'\n\tAverage trace length: {np.mean(trace_lens):.2f}±{np.std(trace_lens):.2f}')

        # get SGT embedding for the traces
        logging.info('========================================')
        logging.info(f'Getting SGT embeddings for {num_traces} traces and {num_features} features...')

        sgt = SGT(kappa=self.kappa,
                  length_sensitive=self.length_sensitive,
                  num_processes=None if self.num_processes < 1 else self.num_processes, verbose=True)
        embeddings = sgt.fit([t for idx, t in traces])  # shape: (num_traces, num_feat_pairs)

        # write file with all symbols in the alphabet
        with open(os.path.join(self.output_dir, 'alphabet.txt'), 'w') as fp:
            fp.write('\n'.join(map(str, sgt.alphabet)))

        # filters embedding features
        feature_labels = [f'{pair[0]}{SGT_FEAT_PAIR_SEP_STR}{pair[1]}' for pair in sgt.features]
        embed_df = pd.DataFrame(embeddings, columns=feature_labels)
        embed_df = embed_df.loc[:, (embed_df != embed_df.iloc[0]).any()]  # remove constant columns
        num_features = len(embed_df.columns)
        if num_features < len(feature_labels):
            logging.info(f'Removed {len(feature_labels) - num_features} features with constant values, '
                         f'total is now {num_features}')
            feature_labels = embed_df.columns

        embed_df[TRACE_IDX_COL] = [idx for idx, t in traces]  # add trace idx column
        embed_df = embed_df[[TRACE_IDX_COL]+list(feature_labels)]
        return embed_df
