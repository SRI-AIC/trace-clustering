import logging
import numpy as np
import pandas as pd
import itertools as it
import scipy.spatial.distance as distances
import sklearn.metrics.pairwise as sk_metrics
from typing import Optional, Callable, Union
from fastdtw import dtw
from enum import IntEnum
from trace_clustering.bin.split_data import TRACE_IDX_COL
from trace_clustering.util.mp import run_parallel

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

DTW_DIST_FUNCTION = distances.cosine


class DistanceType(IntEnum):
    euclidean = 1
    manhattan = 2
    cosine = 3
    minkowski = 4
    dtw = 5

    # https://stackoverflow.com/a/55500795
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return DistanceType[s.lower()]
        except KeyError:
            return s


def get_distances(distance_type: DistanceType,
                  embed_df: pd.DataFrame,
                  num_processes: Optional[int] = -1,
                  **kwargs) -> np.ndarray:
    """
    Get the distances between the given embeddings using the specified distance metric.
    :param DistanceType distance_type: the distance metric to use.
    :param pd.DataFrame embed_df: a dataframe containing the embedding representation for each trace.
    :param int num_processes: number of processes for parallel processing. Value < 1 uses all available cpus.
    :param dict kwargs: a dictionary containing additional options for the distance metric computation.
    :rtype: np.ndarray
    :return: an array of shape (num_embeddings, num_embeddings) containing the pairwise distances between the
    give embeddings.
    """
    embeddings = embed_df.loc[:, embed_df.columns != TRACE_IDX_COL].values
    if distance_type == DistanceType.euclidean:
        return sk_metrics.euclidean_distances(embeddings)
    if distance_type == DistanceType.manhattan:
        return sk_metrics.manhattan_distances(embeddings)
    if distance_type == DistanceType.cosine:
        return sk_metrics.cosine_distances(embeddings)
    if distance_type == DistanceType.minkowski:
        return distances.cdist(embeddings, embeddings, 'minkowski', p=kwargs['p_norm'])
    if distance_type == DistanceType.dtw:
        dist_func = kwargs.get('dtw_dist', None)
        if str(dist_func).isnumeric():
            dist_func = int(str(dist_func))  # p-norm
        elif hasattr(distances, str(dist_func)):
            dist_func = getattr(distances, str(dist_func))  # scipy function
        else:
            dist_func = DTW_DIST_FUNCTION  # default
        logging.info(f'Using {dist_func} as DTW distance function')
        return get_dtw_distances(embed_df, dist_func, num_processes)
    raise NotImplementedError(f'Cannot computed distance of type: {distance_type}')


def get_dtw_distances(embed_df: pd.DataFrame,
                      dist_func: Union[Callable, int],
                      num_processes: Optional[int] = -1) -> np.ndarray:
    """
    Gets the pairwise distances between all traces using dynamic time warping (DTW).
    :param pd.Dataframe embed_df: the dataframe containing the sequence of features for each trace.
    :param int num_processes: number of processes for parallel processing. None or value < 1 uses all available cpus.
    :param function or int dist_func: the distance function used to compare each component of the trace during DTW.
    :rtype: np.ndarray
    :return: an array of shape (num_traces, num_traces) containing the pairwise distances between the traces.
    """
    num_traces = len(embed_df[TRACE_IDX_COL].unique())
    traces = [(t, df.loc[:, embed_df.columns != TRACE_IDX_COL].values) for t, df in embed_df.groupby(TRACE_IDX_COL)]

    # first correct all NaNs
    for t, trace in traces:
        trace[np.isnan(trace)] = -1

    # get all combinations
    args = []
    for trace_pair in it.combinations(traces, 2):
        (t1_idx, trace1), (t2_idx, trace2) = trace_pair
        args.append((t1_idx, trace1, t2_idx, trace2, dist_func))

    # compute DTW distances for all combinations
    dists = run_parallel(_compute_dtw, args, num_processes, use_tqdm=True)

    # add results to distance matrix
    dist_matrix = np.zeros((num_traces, num_traces))
    for t1_idx, t2_idx, (dist, path) in dists:
        dist_matrix[t1_idx, t2_idx] = dist_matrix[t2_idx, t1_idx] = dist

    return dist_matrix


def _compute_dtw(t1_idx: int, trace1: np.ndarray, t2_idx: int, trace2: np.ndarray,
                 dist_func: Union[Callable, int]):
    return t1_idx, t2_idx, dtw(trace1, trace2, dist=dist_func)
