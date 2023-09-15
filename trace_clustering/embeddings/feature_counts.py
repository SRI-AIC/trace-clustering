import logging
import tqdm
import numpy as np
import pandas as pd
from typing import Tuple, List, Set, Optional
from trace_clustering.bin.split_data import TIMESTEP_COL, TRACE_IDX_COL
from trace_clustering.embeddings import EmbeddingAlgorithm

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class FeatureCountsEmbedding(EmbeddingAlgorithm):
    """
    An algorithm that computes a trace embedding with the counts for all feature values, possibly time-discounted.
    """

    def __init__(self,
                 output_dir: str,
                 features_filter: Optional[Set[str]],
                 discount: float = 1.,
                 num_processes: int = None,
                 filter_constant: bool = True,
                 max_timesteps: int = -1):
        """
        Initializes the embedding extraction algorithm.
        :param str output_dir: the output in which to save algorithm's results.
        :param set[str] features_filter: the names of the features to be used for clustering.
        :param float discount: the time discount factor to be applied to feature counts, i.e., such that features
        appearing earlier in a trace have a higher weight than those appearing later in the trace.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param bool filter_constant: whether to remove features whose value is constant across all traces in the dataset.
        """
        super().__init__(output_dir, features_filter, num_processes, filter_constant, max_timesteps)
        self.discount = discount

    def get_embeddings(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get embeddings from the given dataset of traces.
        :param pd.DataFrame traces_df: the pandas dataframe containing the high-level features for all traces/episodes.
        :rtype: Tuple[np.ndarray, List[str]]
        :return: a pandas dataframe containing the features (columns) for each trace.
        """
        # filter features
        features, traces_df = self._pre_process_traces(traces_df)
        num_features = len(features)

        logging.info('========================================')
        logging.info(f'Getting (discounted) counts for all {num_features} features...')
        # add discount factor for each step
        traces_df[TIMESTEP_COL] = self.discount ** (traces_df[TIMESTEP_COL] - 1)  # -1 because t=0 is removed

        # processes features
        feature_labels = []
        feat_series = []
        for feature in tqdm.tqdm(features):
            dtype = traces_df[feature].dtype
            if dtype in [bool, object]:
                # if categorical, get one-hot encodings of the features
                traces_df[feature] = traces_df[feature].astype(str)  # set as string
                for feat_val in traces_df[feature].unique():
                    feat_col = f'{feature}={feat_val}'
                    feat_series.append(pd.Series(
                        pd.to_numeric(traces_df[feature] == feat_val) * traces_df[TIMESTEP_COL], name=feat_col))
                    feature_labels.append(feat_col)
            elif dtype in [int, float]:
                # otherwise just get discounted values, replacing in the dataframe
                traces_df[feature] = traces_df[feature] * traces_df[TIMESTEP_COL]
                feature_labels.append(feature)
            else:
                logging.warning(f'Could not process feature "{feature}", invalid type: {dtype}')

        traces_df = pd.concat([traces_df] + feat_series, axis=1)

        logging.info('========================================')
        logging.info('Organizing data by traces...')
        traces = [(idx, df[feature_labels].values) for idx, df in tqdm.tqdm(traces_df.groupby(TRACE_IDX_COL))]

        # sanity check, see if episodes align with traces indexes
        if not all(idx == t[0] for idx, t in enumerate(traces)):
            traces = sorted(traces, key=lambda x: x[0])

        num_traces = len(traces)
        trace_lens = [len(t) for idx, t in traces]
        logging.info(f'Extracted traces for {num_features} features from the dataset'
                     f'\n\tTotal {num_traces} traces (episodes) per feature'
                     f'\n\tAverage trace length: {np.mean(trace_lens):.2f}±{np.std(trace_lens):.2f}')

        # get embedding for the traces
        logging.info('========================================')
        logging.info(f'Getting feature count embeddings for {num_traces} traces and {num_features} features...')

        feature_counts = np.array([np.nansum(t, axis=0) for idx, t in traces])  # shape: (num_traces, num_feat_vals))
        embed_df = pd.DataFrame(feature_counts, columns=feature_labels)
        embed_df[TRACE_IDX_COL] = [idx for idx, t in traces]  # add trace idx column
        embed_df = embed_df[[TRACE_IDX_COL] + feature_labels]
        return embed_df
