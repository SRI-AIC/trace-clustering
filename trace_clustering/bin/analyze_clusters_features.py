import logging
import os
import argparse
import shutil
import tqdm
import pandas as pd
import numpy as np
import itertools as it
from trace_clustering.util.cmd_line import str2bool, save_args
from trace_clustering.util.io import create_clear_dir
from trace_clustering.util.logging import change_log_handler
from trace_clustering.bin.cluster_traces import CLUSTER_ID_COL, TRACE_IDX_COL

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Loads a file containing feature embeddings for a set of traces and a file with the composition of ' \
           'clusters, computes the mean embedding vector for each cluster, and then calculates the mean pairwise ' \
           'squared error between the embedding features that determines which embedding features contribute the ' \
           'most/least for the between-clusters distances.'

NUM_TOP_FEATURES = 20


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_file', '-ef', type=str, required=True,
                        help='File containing the embeddings for all traces.')
    parser.add_argument('--clusters_file', '-cf', type=str, required=True,
                        help='CSV file containing the clusters for each trace.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save results')

    parser.add_argument('--top', type=int, default=NUM_TOP_FEATURES,
                        help='Number of embedding features that contribute the most for between-clusters distances'
                             'to print to the screen.')

    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes for parallel processing. Value < 1 uses all available cpus.')
    parser.add_argument('--format', '-f', type=str, default='png', help='Format of resulting images.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    # create output
    create_clear_dir(args.output, args.clear)
    save_args(args, os.path.join(args.output, 'args.json'))
    change_log_handler(os.path.join(args.output, 'analyze-clusters-features.log'), args.verbosity)

    # loads embeddings info
    logging.info('========================================')
    logging.info(f'Loading embeddings for all traces from: {args.embeddings_file}...')
    if not os.path.isfile(args.embeddings_file):
        raise ValueError(f'Embeddings file does not exist: {args.embeddings_file}!')
    embed_df = pd.read_pickle(args.embeddings_file)
    logging.info(f'Got {embed_df.shape[0]} embeddings for {embed_df.shape[1]} embedding features')

    # loads clusters file
    logging.info('========================================')
    logging.info(f'Loading clusters info from: {args.clusters_file}...')
    if not os.path.isfile(args.clusters_file):
        raise ValueError(f'Clusters file does not exist: {args.clusters_file}!')
    shutil.copy(args.clusters_file, os.path.join(args.output, os.path.basename(args.clusters_file)))
    clusters_df = pd.read_csv(args.clusters_file)
    if len(clusters_df) != embed_df.shape[0]:
        raise ValueError(f'Number of traces ({len(clusters_df)}) does not coincide with '
                         f'number of embeddings ({embed_df.shape[0]})!')
    logging.info(f'Loaded {len(clusters_df[CLUSTER_ID_COL].unique())} clusters from: {args.clusters_file}...')

    # gets embeddings for each cluster
    logging.info('========================================')
    logging.info('Calculating embeddings for each cluster...')
    cluster_embeds = {}
    for cluster, cluster_df in tqdm.tqdm(clusters_df.groupby(CLUSTER_ID_COL)):
        cluster_embeds[cluster] = embed_df.loc[cluster_df[TRACE_IDX_COL]].values

    # embed_feature_names = [re.sub(r'(\w+)=False', r'~\1', f.replace('=True', '')) for f in embed_df.columns]
    embed_feature_names = embed_df.columns
    mean_embed_df = pd.DataFrame.from_dict({c: np.mean(embeds, axis=0) for c, embeds in cluster_embeds.items()},
                                           orient='index', columns=embed_feature_names)
    mean_embed_df.index.set_names('Cluster', inplace=True)
    mean_embed_df.sort_index(inplace=True)
    file_path = os.path.join(args.output, 'mean-cluster-embeds.csv')
    logging.info(f'Saving mean cluster embedding features to: {file_path}...')
    mean_embed_df.to_csv(file_path)

    # gets pairwise component distances
    logging.info('========================================')
    logging.info('Calculating inter-cluster pairwise embedding distances...')
    clusters = list(cluster_embeds.keys())
    num_clusters = len(clusters)
    mean_embed_diffs = 0.
    embed_count = 0
    for i in range(num_clusters):
        logging.info(f'Processing cluster {i}...')
        c_i = clusters[i]
        len_c_i = len(cluster_embeds[c_i])
        for j in range(i + 1, num_clusters):
            c_j = clusters[j]
            len_c_j = len(cluster_embeds[c_j])
            for k, l in tqdm.tqdm(it.product(range(len_c_i), range(len_c_j)), total=len_c_i * len_c_j):
                if k == l:
                    continue
                dist = (cluster_embeds[c_i][k] - cluster_embeds[c_j][l]) ** 2
                mean_embed_diffs = (mean_embed_diffs * embed_count + dist) / (embed_count + 1)
                embed_count += 1

    # gets mean component distances across all pairs of all clusters
    logging.info('========================================')
    feat_diffs_df = pd.DataFrame(zip(embed_feature_names, mean_embed_diffs),
                                 columns=['Embedding Feature', 'Mean Difference'])
    feat_diffs_df.sort_values('Mean Difference', ascending=False, inplace=True)
    logging.info(f'Top-{args.top} embedding features:')
    logging.info(feat_diffs_df.iloc[:args.top])

    logging.info('========================================')
    file_path = os.path.join(args.output, 'embed-feat-diffs.csv')
    logging.info(f'Saving mean embedding features differences to: {file_path}...')
    feat_diffs_df.to_csv(file_path, index=False)

    logging.info('Done!')


if __name__ == '__main__':
    main()
