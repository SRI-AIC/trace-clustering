import logging
import os
import argparse
import pandas as pd
import numpy as np
from trace_clustering.bin.split_data import TRACE_IDX_COL
from trace_clustering.clustering_distances import DistanceType, get_distances
from trace_clustering.util.cmd_line import str2bool, save_args
from trace_clustering.util.clustering import hopkins_statistic
from trace_clustering.util.io import create_clear_dir, get_file_changed_extension
from trace_clustering.util.logging import change_log_handler
from trace_clustering.util.plot import plot_matrix, dummy_plotly

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Loads a file containing feature embeddings for a set of traces and extracts distances' \
           'between all traces for subsequent clustering. '

DISTANCES_FILE = 'trace-distances.npz'
DIST_MATRIX_PALETTE = 'Viridis'


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings-file', '-e', type=str, required=True,
                        help='File containing the embeddings for all traces.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save results')

    parser.add_argument('--distance-metric', '-dm', type=DistanceType.argparse,
                        default=DistanceType.cosine, choices=list(DistanceType),
                        help='Distance metric used to compute distances between trace embeddings. '
                             'See options in `trace_clustering.clustering_distances.DistanceType`.')
    parser.add_argument('--p_norm', type=int, default=2,
                        help='[minkowsky] The p-norm to apply for Minkowski distance computation.')
    parser.add_argument('--dtw_dist', type=str, default=None,
                        help='[dtw] The distance function used to compare each component of the trace during DTW.')

    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes for parallel processing. Value < 1 uses all available cpus.')
    parser.add_argument('--format', '-f', type=str, default='png', help='Format of resulting images.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    # create output
    create_clear_dir(args.output, args.clear)
    save_args(args, os.path.join(args.output, 'args.json'))
    change_log_handler(os.path.join(args.output, 'get-distances.log'), args.verbosity)

    # loads embeddings info
    logging.info('========================================')
    logging.info(f'Loading embeddings for all traces from: {args.embeddings_file}...')
    if not os.path.isfile(args.embeddings_file):
        raise ValueError(f'Embeddings file does not exist: {args.embeddings_file}!')
    embed_df = pd.read_pickle(args.embeddings_file)
    logging.info(f'Got {embed_df.shape[0]} trace embeddings for {embed_df.shape[1] - 1} embedding features')

    # gets mean distances between all traces based on embeddings
    logging.info('========================================')
    num_traces = len(embed_df[TRACE_IDX_COL].unique())
    logging.info(f'Calculating pairwise distances for {num_traces} traces '
                 f'using "{args.distance_metric.name}" distance between embeddings...')
    distances = get_distances(args.distance_metric, embed_df, args.processes, **vars(args))

    logging.info('========================================')

    # save distances as confusion matrix plot
    file_path = os.path.join(args.output, get_file_changed_extension(DISTANCES_FILE, args.format))
    logging.info(f'Saving confusion matrix plot to: {file_path}...')
    dummy_plotly()  # to hide import msg
    plot_matrix(distances, 'Traces Distances', file_path,
                save_csv=False, palette=DIST_MATRIX_PALETTE, show_values=False,
                symmetrical=True, width=660, height=600)

    # save distances matrix to compressed numpy file
    file_path = os.path.join(args.output, DISTANCES_FILE)
    logging.info(f'Saving distances (shape: {distances.shape}) to: {file_path}...')
    np.savez_compressed(file_path, distances)

    if len(embed_df) == num_traces:
        logging.info('========================================')
        logging.info('Computing Hopkins statistic from the embeddings...')
        h = hopkins_statistic(embed_df.loc[:, embed_df.columns != TRACE_IDX_COL].values)
        logging.info(f'\tH={h:.2f}')
        if h <= 0.3:
            logging.info('\tTrace data is regularly spaced')
        elif 0.45 <= h <= 0.55:
            logging.info('\tTrace data is random')
        elif h > 0.75:
            logging.info('\tTrace data has a high tendency to cluster')

    logging.info('Done!')


if __name__ == '__main__':
    main()
