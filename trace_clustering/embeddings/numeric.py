import logging
import numpy as np
import pandas as pd
import tqdm
from typing import Set, Optional

from trace_clustering.bin.split_data import TRACE_IDX_COL
from trace_clustering.embeddings import EmbeddingAlgorithm

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class NumericEmbedding(EmbeddingAlgorithm):
    """
    An embedding that converts all features to numeric and passes through the raw feature traces.
    To be used with a distance function comparing traces (sequences of features) rather than a vector representation
    of traces.
    """

    def __init__(self,
                 output_dir: str,
                 features_filter: Optional[Set[str]],
                 num_processes: int = None,
                 filter_constant: bool = True,
                 max_timesteps: int = -1):
        """
        Initializes the embedding extraction algorithm.
        :param str output_dir: the output in which to save algorithm's results.
        :param set[str] features_filter: the names of the features to be used for clustering.
        appearing earlier in a trace have a higher weight than those appearing later in the trace.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param bool filter_constant: whether to remove features whose value is constant across all traces in the dataset.
        """
        super().__init__(output_dir, features_filter, num_processes, filter_constant, max_timesteps)

    def get_embeddings(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get embeddings from the given dataset of traces.
        :param pd.DataFrame traces_df: the pandas dataframe containing the high-level features for all traces/episodes.
        :rtype:  pd.DataFrame
        :return: a pandas dataframe containing the features (columns) for each trace.
        """
        # filter features
        features, traces_df = self._pre_process_traces(traces_df)
        num_features = len(features)

        feature_labels = []
        feat_series = []
        for feature in tqdm.tqdm(features):
            dtype = traces_df[feature].dtype
            if dtype == object:
                # if categorical, get one-hot encodings of the features
                traces_df[feature] = traces_df[feature].astype(str)  # set as string
                for feat_val in traces_df[feature].unique():
                    feat_col = f'{feature}={feat_val}'
                    feat_series.append(pd.Series((traces_df[feature] == feat_val).astype(np.uint8), name=feat_col))
                    feature_labels.append(feat_col)
            elif dtype == bool:
                traces_df[feature] = traces_df[feature].astype(np.uint8)  # set as int
                feature_labels.append(feature)
            elif dtype in [int, float]:
                feature_labels.append(feature)  # otherwise don't change feature
            else:
                logging.warning(f'Could not process feature "{feature}", invalid type: {dtype}')

        embed_df = pd.concat([traces_df] + feat_series, axis=1)
        embed_df = embed_df[[TRACE_IDX_COL] + feature_labels]

        logging.info('========================================')
        logging.info(f'Returning {num_features} raw features (size: {embed_df.shape})...')
        return embed_df


class MeanEmbedding(NumericEmbedding):
    """
    An embedding that converts all features to numeric and passes through the mean feature value for each trace.
    """

    def __init__(self, output_dir: str,
                 features_filter: Optional[Set[str]],
                 num_processes: int = None,
                 filter_constant: bool = True,
                 max_timesteps: int = -1):
        super().__init__(output_dir, features_filter, num_processes, filter_constant, max_timesteps)

    def get_embeddings(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        embed_df = super().get_embeddings(traces_df)
        return embed_df.groupby(TRACE_IDX_COL).mean().reset_index()  # just compute the mean
