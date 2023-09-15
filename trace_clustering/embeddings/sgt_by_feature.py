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


class SGTByFeatureEmbedding(EmbeddingAlgorithm):
    """
    An algorithm that computes a trace embedding using SGT that finds temporal relationships between feature values
    *within* each high-level feature, i.e., does not compute across-feature relationships. Pairs of feature values for
    each feature are then concatenated in the final embedding of a trace.
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

    def get_embeddings(self, traces: pd.DataFrame) -> pd.DataFrame:
        """
        Get embeddings from the given dataset of traces.
        :param pd.DataFrame traces: the pandas dataframe containing the high-level features for all traces/episodes.
        :rtype: Tuple[np.ndarray, List[str]]
        :return: a pandas dataframe containing the features (columns) for each trace.
        """

        # filter features
        features, traces = self._pre_process_traces(traces)
        traces[features] = traces[features].astype(str)  # ensure categorical features
        num_features = len(features)

        logging.info('========================================')
        logging.info('Organizing traces by feature...')
        feat_traces = {}
        trace_idxs = []
        trace_lens = []
        for idx, df in tqdm.tqdm(traces.groupby(TRACE_IDX_COL)):
            trace_idxs.append(idx)
            trace_lens.append(len(df))
            # extract traces (sequences of values) for each feature
            for feature in features:
                if feature not in feat_traces:
                    feat_traces[feature] = []
                feat_traces[feature].append(df[feature].values)

        num_traces = len(trace_lens)
        logging.info(f'Extracted traces for {num_features} features from the dataset'
                     f'\n\tTotal {num_traces} traces (episodes) per feature'
                     f'\n\tAverage trace length: {np.mean(trace_lens):.2f}±{np.std(trace_lens):.2f}')

        # get SGT embedding for each feature
        logging.info('========================================')
        logging.info(f'Getting embeddings for {num_traces} traces and {len(feat_traces)} features...')
        embeddings = []
        feature_labels = []
        out_dir = os.path.join(self.output_dir, 'embeddings')
        for feature, traces in tqdm.tqdm(feat_traces.items()):
            logging.info(f'Processing feature "{feature}"...')

            # gets embedding via SGT
            sgt = SGT(kappa=self.kappa,
                      length_sensitive=self.length_sensitive,
                      num_processes=None if self.num_processes < 1 else self.num_processes, verbose=True)
            embeds = sgt.fit(traces)  # shape: (num_traces, num_feat_pairs)
            embeddings.append(embeds)
            feature_labels.extend(f'{feature}{SGT_FEAT_VAL_SEP_STR}{pair[0]}'
                                  f'{SGT_FEAT_PAIR_SEP_STR}{feature}{SGT_FEAT_VAL_SEP_STR}{pair[1]}'
                                  for pair in sgt.features)

            # saves feature embeddings for each sequence to file
            os.makedirs(out_dir, exist_ok=True)
            file_name = os.path.join(out_dir, f'{feature.lower()}.csv')
            sgt.get_dataframe(embeds).to_csv(file_name, index=False)

            # saves summary stats to file
            stats = pd.DataFrame(embeds).describe()
            file_name = os.path.join(out_dir, f'{feature.lower()}-stats.csv')
            stats.to_csv(file_name, index=True, header=False)

        # concatenates embeddings for all features
        embeddings = np.hstack(embeddings)  # shape: (num_traces, num_all_feat_pairs)

        # filters embedding features
        embed_df = pd.DataFrame(embeddings, columns=feature_labels)
        embed_df = embed_df.loc[:, (embed_df != embed_df.iloc[0]).any()]  # remove constant columns
        num_features = len(embed_df.columns)
        if num_features < len(feature_labels):
            logging.info(f'Removed {len(feature_labels) - num_features} features with constant values, '
                         f'total is now {num_features}')
            feature_labels = embed_df.columns

        embed_df[TRACE_IDX_COL] = trace_idxs  # add trace idx column
        embed_df = embed_df[[TRACE_IDX_COL] + list(feature_labels)]
        return embed_df
