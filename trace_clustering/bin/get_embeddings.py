import argparse
import json
import logging
import os
import pandas as pd
import shutil
from sklearn import preprocessing
from typing import Set, Optional

from trace_clustering.bin.split_data import TRACE_IDX_COL
from trace_clustering.embeddings import EmbeddingType, EmbeddingAlgorithm
from trace_clustering.embeddings.feature_counts import FeatureCountsEmbedding
from trace_clustering.embeddings.numeric import NumericEmbedding, MeanEmbedding
from trace_clustering.embeddings.sgt_by_feature import SGTByFeatureEmbedding
from trace_clustering.embeddings.sgt_feature_sets import SGTFeatureSetEmbedding
from trace_clustering.util.cmd_line import save_args, str2bool
from trace_clustering.util.io import create_clear_dir
from trace_clustering.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Loads a file containing feature traces (sequences) and extracts embeddings' \
           '(a vector representation for each trace) for subsequent trace clustering.'

EMBEDDINGS_FILE = 'embeddings.pkl.gz'


def _get_embedding_alg(args: argparse.Namespace, features_filter: Optional[Set[str]]) -> EmbeddingAlgorithm:
    # select embedding algorithm from cmd-line args
    if args.embedding_algorithm == EmbeddingType.sgt_by_feature:
        return SGTByFeatureEmbedding(args.output, features_filter, args.kappa, args.lengthsensitive,
                                     args.processes, args.filter_constant, args.time_max)
    if args.embedding_algorithm == EmbeddingType.sgt_feature_sets:
        return SGTFeatureSetEmbedding(args.output, features_filter, args.kappa, args.lengthsensitive,
                                      args.processes, args.filter_constant, args.time_max)
    if args.embedding_algorithm == EmbeddingType.feature_counts:
        return FeatureCountsEmbedding(args.output, features_filter,
                                      args.discount, args.processes, args.filter_constant, args.time_max)
    if args.embedding_algorithm == EmbeddingType.numeric:
        return NumericEmbedding(args.output, features_filter, args.processes, args.filter_constant, args.time_max)
    if args.embedding_algorithm == EmbeddingType.mean:
        return MeanEmbedding(args.output, features_filter, args.processes, args.filter_constant, args.time_max)

    raise NotImplementedError(f'Cannot create embedding algorithm of type: {args.embedding_algorithm}')


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Pandas dataset pickle file containing high-level feature sequences (traces).')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save results')
    parser.add_argument('--features-filter', type=str, default=None,
                        help='Path to file containing the names of the features to be used for clustering.')
    parser.add_argument('--filter-constant', type=str2bool, default='True',
                        help='Whether to remove features whose value is constant across all traces in the dataset.')
    parser.add_argument('--time-max', '-t', type=int, default=-1,
                        help='Maximum number of timesteps in a trace used for embedding calculation.'
                             '`-1` will use all timesteps.')

    parser.add_argument('--embedding-algorithm', '-ea', type=EmbeddingType.argparse,
                        default=EmbeddingType.sgt_feature_sets, choices=list(EmbeddingType),
                        help='Embedding algorithm used to compute distances between traces. '
                             'See options in `trace_clustering.embeddings.EmbeddingType`.')

    parser.add_argument('--kappa', '-k', type=float, default=1,
                        help='[sgt] Tuning parameter for SGT, kappa > 0, to change the extraction of '
                             'long-term dependency. Higher the value the lesser the long-term '
                             'dependency captured in the embedding. Typical values for kappa are 1, 5, 10.')
    parser.add_argument('--lengthsensitive', '-ls', type=str2bool,
                        help='[sgt] This is set to true if the embedding produced by SGT '
                             'should have the information of the length of the sequence. '
                             'If set to false then the embedding of two sequences with '
                             'similar pattern but different lengths will be the same.'
                             'False is similar to length-normalization.')
    parser.add_argument('--discount', type=float, default=1,
                        help='[feature_count] The time discount factor to be applied to '
                             'feature counts, i.e., such that features appearing earlier '
                             'in a trace have a higher weight than those appearing later '
                             'in the trace.')

    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes for parallel processing. Value < 1 uses all available cpus.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    if args.input is None or not os.path.isfile(args.input):
        raise ValueError(f'Invalid input file provided: {args.input}!')
    create_clear_dir(args.output, args.clear)
    save_args(args, os.path.join(args.output, 'args.json'))
    change_log_handler(os.path.join(args.output, 'get-embeddings.log'), args.verbosity)

    logging.info('========================================')

    # checks features filter file
    features_filter = None
    if args.features_filter is not None and os.path.isfile(args.features_filter):
        with open(args.features_filter, 'r') as fp:
            features_filter = set(json.load(fp))
        logging.info(f'Loaded {len(features_filter)} features from filter file: {args.features_filter}')
        shutil.copy(args.features_filter, os.path.join(args.output, os.path.basename(args.features_filter)))
    else:
        logging.info('No feature filter will be used')

    # loads dataset from input file
    logging.info('========================================')
    logging.info(f'Loading dataset from {args.input}...')
    traces_df = pd.read_pickle(args.input)
    logging.info(f'Loaded {len(traces_df)} entries and {len(traces_df.columns)} columns')

    # select features and get embeddings for each trace / episode
    logging.info('========================================')
    logging.info(f'Getting embeddings using "{args.embedding_algorithm.name}"...')
    embeds_alg = _get_embedding_alg(args, features_filter)
    embed_df = embeds_alg.get_embeddings(traces_df)
    feat_cols = embed_df.columns != TRACE_IDX_COL
    logging.info(
        f'Got {len(embed_df[TRACE_IDX_COL].unique())} trace embeddings for {sum(feat_cols)} embedding features')

    # normalize embeddings by scaling the values for each feature
    embed_df.loc[:, feat_cols] = preprocessing.MinMaxScaler().fit_transform(embed_df.loc[:, feat_cols])

    # save embeddings to compressed file
    logging.info('========================================')
    file_path = os.path.join(args.output, EMBEDDINGS_FILE)
    logging.info(f'Saving CSV file with all traces embeddings to: {file_path}...')
    embed_df.to_pickle(file_path, compression='gzip')

    logging.info('Done!')


if __name__ == '__main__':
    main()
