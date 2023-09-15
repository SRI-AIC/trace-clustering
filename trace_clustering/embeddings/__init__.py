import json
import logging
import os
import numpy as np
import pandas as pd
import tqdm
from enum import IntEnum
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Set, Optional
from trace_clustering.bin.split_data import TIMESTEP_COL, FEATURES_START_IDX

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

SGT_FEAT_PAIR_SEP_STR = '=>'
SGT_FEAT_VAL_SEP_STR = '='


class EmbeddingType(IntEnum):
    sgt_by_feature = 1
    sgt_feature_sets = 2
    feature_counts = 3
    numeric = 4
    mean = 5

    # https://stackoverflow.com/a/55500795
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return EmbeddingType[s.lower()]
        except KeyError:
            return s


class EmbeddingAlgorithm(object, metaclass=ABCMeta):
    """
    Represents an algorithm that creates an embedding (vector representation) for a trace/episode (sequence of feature
    values).
    """

    def __init__(self, output_dir: str,
                 features_filter: Optional[Set[str]],
                 num_processes: int = None,
                 filter_constant: bool = True,
                 max_timesteps: int = -1):
        """
        Initializes the embedding extraction algorithm.
        :param str output_dir: the output in which to save algorithm's results.
        :param set[str] features_filter: the names of the features to be used for clustering.
        :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
        :param bool filter_constant: whether to remove features whose value is constant across all traces in the dataset.
        :param int max_timesteps: maximum number of timesteps in a trace used for embedding calculation.
        """
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.features_filter = features_filter
        self.filter_constant = filter_constant
        self.max_timesteps = max_timesteps

    @abstractmethod
    def get_embeddings(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get embeddings from the given dataset of traces.
        :param pd.DataFrame traces_df: the pandas dataframe containing the high-level features for all traces/episodes.
        :rtype:  pd.DataFrame
        :return: a pandas dataframe containing the features (columns) for each trace.
        """
        pass

    def _pre_process_traces(self, traces_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Pre-processes the dataset of traces by dropping the first timestep of every trace and then removing the features
        whose value does not change across *all* traces. Also prints the remaining features to a text file.
        :param pd.DataFrame traces_df: the pandas dataframe containing the high-level features for all traces/episodes.
        :return: a tuple containing the filtered column names and the pre-processed traces dataset.
        """
        logging.info('========================================')
        logging.info('Processing dataset...')
        traces_df = traces_df.copy()  # make copy as we're editing columns
        features = list(traces_df.columns.values[FEATURES_START_IDX:])
        logging.info(f'There is a total of {len(features)} features available')
        if self.features_filter is not None and len(self.features_filter) > 0:
            features = list(self.features_filter.intersection(features))
            logging.info(f'Considering {len(features)} features after filtering')

        logging.info(f'Removing first timestep of traces' +
                     '...' if self.max_timesteps == -1 else f' and all timesteps above {self.max_timesteps}...')
        ts_filter = (traces_df[TIMESTEP_COL] == 0) | \
                    (False if self.max_timesteps == -1 else traces_df[TIMESTEP_COL] > self.max_timesteps)
        traces_df.drop(traces_df[ts_filter].index, inplace=True)  # delete first and > time_max timesteps of traces

        if self.filter_constant:
            # remove constant features
            logging.info('Filtering constant features...')
            new_features = [feature for feature in tqdm.tqdm(features) if len(traces_df[feature].unique()) > 1]
            new_features.sort(key=lambda x: features.index(x))
            logging.info(f'Dropped {len(features) - len(new_features)} constant features from the dataset')
            const_features = list(set(features) - set(new_features))
            const_features.sort(key=lambda x: features.index(x))
            with open(os.path.join(self.output_dir, 'constants.json'), 'w') as fp:
                json.dump(const_features, fp, indent=4)

            # remove duplicate features
            logging.info('Finding duplicated features...')
            duplicate_features = _find_duplicate_features(traces_df, new_features)
            duplicate_features.sort(key=lambda x: features.index(x[0]))
            num_duplicates = sum(len(feature_cluster) for feature_cluster in duplicate_features)
            logging.info(f'Found {num_duplicates} duplicated features from the dataset')
            with open(os.path.join(self.output_dir, 'duplicates.json'), 'w') as fp:
                json.dump(duplicate_features, fp, indent=4)

            # # drops duplicates
            # merged_features = []
            # for feature_cluster in duplicate_features:
            #     cluster_id = ' | '.join(feature_cluster)
            #     traces_df[cluster_id] = traces_df[feature_cluster[0]]  # add new column
            #     merged_features.append(cluster_id)
            #     for f in feature_cluster:
            #         new_features.remove(f)
            #
            # logging.info(f'Dropped {len(merged_features)} duplicated features from the dataset')
            dropped_features = list(set(features) - set(new_features))
            # filtered_features += merged_features
            dropped_features.sort(key=lambda x: features.index(x))
            with open(os.path.join(self.output_dir, 'dropped.json'), 'w') as fp:
                json.dump(dropped_features, fp, indent=4)
            features = new_features

        # write files with features
        with open(os.path.join(self.output_dir, 'features.json'), 'w') as fp:
            json.dump(features, fp, indent=4)

        return features, traces_df


def _find_duplicate_features(traces_df: pd.DataFrame, filtered_features: List[str]) -> List[List[str]]:
    # finds pairs of features for which all values are equal
    duplicates = {}
    for i, fi in enumerate(tqdm.tqdm(filtered_features)):
        for j in range(i + 1, len(filtered_features)):
            fj = filtered_features[j]
            if not traces_df[fi].equals(traces_df[fj]):
                continue
            if fi not in duplicates:
                duplicates[fi] = [fi]
            duplicates[fi].append(fj)
            duplicates[fj] = duplicates[fi]
            break

    # gets unique feature clusters
    duplicate_features = []
    for feature_cluster in duplicates.values():
        exists = False
        for unique_cluster in duplicate_features:
            if feature_cluster == unique_cluster:
                exists = True
                break
        if not exists:
            duplicate_features.append(feature_cluster)
    return duplicate_features
