import logging
import os
import argparse
import pandas as pd
import numpy as np
import tqdm
from sklearn.cluster import AgglomerativeClustering
from trace_clustering.bin.get_distances import DIST_MATRIX_PALETTE
from trace_clustering.bin.split_data import PARTITION_COL, TRACE_IDX_COL, TRACE_ID_COL, TIMESTEP_COL
from trace_clustering.util.clustering import update_clusters, get_sorted_indexes, plot_clustering_distances, \
    plot_clustering_dendrogram, Orientation, get_internal_evaluations, \
    get_external_evaluations
from trace_clustering.util.cmd_line import str2bool, save_args
from trace_clustering.util.data import save_separate_csv_gzip
from trace_clustering.util.io import create_clear_dir
from trace_clustering.util.logging import change_log_handler
from trace_clustering.util.plot import plot_bar, plot_matrix, plot_timeseries, dummy_plotly

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Loads a matrix with pairwise distances between traces (sequences) and clusters them using Hierarchical ' \
           'Agglomerative Clustering (HAC). Loads ground-truth partitions for clustering evaluation.'

CLUSTER_TRACES_FILE = 'cluster-traces.csv'
TRACE_CLUSTERS_FILE = 'trace-clusters.csv'

CLUSTER_ID_COL = 'Cluster'
CLUSTER_COUNT_COL = 'Count'


def _internal_evaluation(clustering: AgglomerativeClustering, dist_matrix: np.ndarray, args: argparse.Namespace):
    # loads embeddings info to perform internal evaluation
    logging.info('========================================')
    logging.info(f'Loading embeddings for all traces from: {args.embeddings_file}...')
    if not os.path.isfile(args.embeddings_file):
        raise ValueError(f'Embeddings file does not exist: {args.embeddings_file}!')
    embed_df = pd.read_pickle(args.embeddings_file)
    logging.info(f'Got {embed_df.shape[0]} embeddings for {embed_df.shape[1]} embedding features')

    # performs internal evaluation using different metrics and num. clusters
    logging.info('========================================')
    sub_dir = os.path.join(args.output, 'internal eval')
    os.makedirs(sub_dir, exist_ok=True)

    max_clusters = args.eval_clusters
    logging.info(f'Performing internal evaluation for up to {max_clusters} clusters, saving results in "{sub_dir}"...')
    seq_embeddings = len(embed_df) > len(embed_df[TRACE_IDX_COL].unique())  # check if more than one embed per trace
    evals = get_internal_evaluations(clustering, max_clusters,
                                     embed_df.values if not seq_embeddings else None, dist_matrix)

    # saves plots for each metric with scores for diff. num clusters
    for metric, scores in evals.items():
        file_path = os.path.join(sub_dir, f'{metric.lower().replace(" ", "-")}.{args.format}')
        df = pd.DataFrame({'Score': scores.values()}, index=pd.Index(scores.keys(), name='Num. Clusters'))
        plot_timeseries(df, metric, file_path, x_label='Num. Clusters', y_label='Score', show_legend=False)


def _external_evaluation(clustering: AgglomerativeClustering, clusters,
                         traces_df: pd.DataFrame, args: argparse.Namespace):
    logging.info('========================================')
    sub_dir = os.path.join(args.output, 'external eval')
    os.makedirs(sub_dir, exist_ok=True)
    logging.info(f'Performing external evaluation, saving results in "{sub_dir}"...')

    # visualize distribution over GT partitions in each cluster
    partitions = list(traces_df[PARTITION_COL].unique())
    logging.info(f'Visualizing distribution over {len(partitions)} ground-truth partitions...')
    for cluster in tqdm.tqdm(sorted(clusters)):
        traces = clusters[cluster]
        partition_props = {p: len([t for t in traces
                                   if np.any(traces_df.loc[traces_df[TRACE_IDX_COL] == t][PARTITION_COL] == p)])
                           for p in partitions}
        total = sum(partition_props.values())

        logging.info(f'Cluster {cluster} has {len(traces)} traces:')
        for p, prop in partition_props.items():
            logging.info(f'\t{int((prop / total) * 100)}%\t{p}')

        plot_bar(partition_props, f'Ground-truth Partitions for Cluster {cluster}',
                 os.path.join(sub_dir, f'cluster-{cluster}.{args.format}'),
                 plot_mean=False, show_legend=False, x_label='Partition', y_label='Count')

    # performs external evaluation using different metrics
    logging.info('========================================')

    # get gt partition labels, see if episodes align with traces indexes
    logging.info('Getting GT labels for all traces...')
    labels_true = [(t, partitions.index(df[PARTITION_COL].values[0]))
                   for t, df in tqdm.tqdm(traces_df.groupby(TRACE_IDX_COL))]
    if not all(idx == t for idx, (t, p) in enumerate(labels_true)):
        labels_true = sorted(labels_true, key=lambda x: x[0])
    labels_true = [p for t, p in labels_true]

    max_clusters = args.eval_clusters
    logging.info(f'Performing external evaluation for up to {max_clusters} clusters...')
    evals = get_external_evaluations(clustering, max_clusters, labels_true)

    # saves plots for each metric with scores for diff. num clusters
    for metric, scores in evals.items():
        file_path = os.path.join(sub_dir, f'{metric.lower().replace(" ", "-")}.{args.format}')
        df = pd.DataFrame({'Score': scores.values()}, index=pd.Index(scores.keys(), name='Num. Clusters'))
        plot_timeseries(df, metric, file_path, x_label='Num. Clusters', y_label='Score', show_legend=False)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--distances', '-d', type=str, required=True,
                        help='Numpy binary file containing the distances matrix between all traces.')
    parser.add_argument('--traces', '-t', type=str, required=True,
                        help='CSV file containing the traces data (features).')
    parser.add_argument('--embeddings_file', '-e', type=str, required=True,
                        help='CSV file containing the embeddings for all traces.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save results')

    parser.add_argument('--linkage', '-l', type=str, default='average',
                        help='Which linkage criterion to use. The linkage criterion '
                             'determines which distance to use between sets of observation. '
                             'The algorithm will merge the pairs of cluster that minimize '
                             'this criterion. One of: "complete", "average", "single".')
    parser.add_argument('--n_clusters', '-n', type=int, default=-1,
                        help='The number of clusters to find. Value of -1 will use distance_threshold.')
    parser.add_argument('--distance_threshold', '-dt', type=float, default=0.025,
                        help='The linkage distance threshold above which, clusters will not be merged.')
    parser.add_argument('--eval_clusters', '-ec', type=int, default=7,
                        help='Maximum number of clusters for which to perform evaluation')

    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes for parallel processing. Value < 1 uses all available cpus.')
    parser.add_argument('--format', '-f', type=str, default='png', help='Format of result images')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    # prepares output dir
    create_clear_dir(args.output, args.clear)
    save_args(args, os.path.join(args.output, 'args.json'))
    change_log_handler(os.path.join(args.output, 'cluster-traces.log'), args.verbosity)

    # loads distances
    logging.info('========================================')
    logging.info(f'Loading distances file from {args.distances}...')
    if not os.path.isfile(args.distances):
        raise ValueError(f'Distances file does not exist: {args.distances}!')
    dist_matrix = np.load(args.distances)
    dist_matrix = dist_matrix[dist_matrix.files[0]]
    if dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError(f'Distances matrix should be a square: {dist_matrix.shape}!')
    logging.info(f'Loaded distances for {dist_matrix.shape[0]} traces')

    # perform trace clustering based on embeddings
    logging.info('========================================')
    logging.info(f'Clustering {dist_matrix.shape[0]} traces via HAC...')
    clustering = AgglomerativeClustering(n_clusters=None if args.n_clusters == -1 else args.n_clusters,
                                         affinity='precomputed',
                                         linkage=args.linkage,
                                         compute_distances=True,
                                         distance_threshold=args.distance_threshold if args.n_clusters == -1 else None)
    clustering.fit(dist_matrix)

    # performs edge/slope detection
    grad = np.gradient(clustering.distances_)
    max_edge = np.argmax(grad)
    logging.info(f'Maximum distance edge discovered at {clustering.distances_[max_edge]:.2f} '
                 f'(n clusters: {len(grad) + 1 - max_edge}).')

    # manually update clusters if distance threshold is lower than expected
    if (clustering.distance_threshold is not None and args.n_clusters == -1 and
            clustering.distances_[max_edge] < clustering.distance_threshold):
        logging.info('Performing automatic num. clusters detection...')
        update_clusters(clustering, clustering.distances_[max_edge])

    logging.info(f'Found {clustering.n_clusters_} clusters at max. distance: {clustering.distance_threshold}')

    # saves clustering results
    logging.info('========================================')
    logging.info('Saving clustering results...')
    dummy_plotly()
    plot_clustering_distances(clustering, os.path.join(args.output, f'clustering-distances.{args.format}'))
    plot_clustering_dendrogram(clustering, 'Clustering Dendrogram',
                               os.path.join(args.output, f'clustering-dendrogram.{args.format}'),
                               orientation=Orientation.bottom)

    # saves distances matrix sorted by cluster
    logging.info('========================================')
    logging.info('Sorting traces according to clustering results...')
    sorted_idxs = get_sorted_indexes(clustering)
    distances = dist_matrix[:, sorted_idxs][sorted_idxs, :]
    file_path = os.path.join(args.output, f'trace-distances.{args.format}')
    logging.info(f'Saving distances confusion matrix plot to: {file_path}...')
    plot_matrix(distances, 'Traces Distances', file_path,
                save_csv=False, palette=DIST_MATRIX_PALETTE, show_values=False,
                symmetrical=True, width=660, height=600)

    # performs internal evaluation
    _internal_evaluation(clustering, dist_matrix, args)

    # loads traces info to get partitions info for external evaluation
    logging.info('========================================')
    logging.info(f'Loading CSV file with all traces data from: {args.traces}...')
    if not os.path.isfile(args.traces):
        raise ValueError(f'Traces info file does not exist: {args.traces}!')
    traces_df = pd.read_pickle(args.traces)

    # gets traces idxs in each cluster
    clusters = {}
    for idx, cluster in enumerate(clustering.labels_):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(idx)

    clusters_df = pd.DataFrame.from_dict({cluster: [len(traces), traces] for cluster, traces in clusters.items()},
                                         orient='index', columns=[CLUSTER_COUNT_COL, TRACE_IDX_COL])
    clusters_df.index.names = [CLUSTER_ID_COL]
    clusters_df.sort_index(inplace=True)
    clusters_df.to_csv(os.path.join(args.output, CLUSTER_TRACES_FILE))

    # performs external evaluation
    _external_evaluation(clustering, clusters, traces_df, args)

    # save each cluster's traces data in a separate CSV file
    logging.info('========================================')
    sub_dir = os.path.join(args.output, 'traces')
    os.makedirs(sub_dir, exist_ok=True)
    logging.info(f'Saving clusters\' data to {sub_dir}...')
    traces_df.rename(columns={PARTITION_COL: CLUSTER_ID_COL}, inplace=True)  # replace GT partition with cluster column
    traces_df[CLUSTER_ID_COL] = -1  # set to dummy value to make sure no trace is missed
    for cluster, traces in tqdm.tqdm(sorted(clusters.items())):
        idxs = traces_df[TRACE_IDX_COL].isin(traces)
        cluster_df = traces_df.loc[idxs].copy()
        cluster_df[CLUSTER_ID_COL] = cluster  # set cluster id column

        # save to single gzip file with multiple csv files inside
        file_path = os.path.join(sub_dir, f'cluster-{cluster}.tar.gz')
        save_separate_csv_gzip(cluster_df, file_path, group_by=TRACE_ID_COL, use_group_filename=True, use_tqdm=False)
        traces_df.loc[idxs, CLUSTER_ID_COL] = cluster  # set cluster id column

    # check if all traces have a cluster
    assert not np.any(traces_df[CLUSTER_ID_COL] == -1), 'Some traces do not have a cluster assigned!'

    # save table with episode id, replay, cluster for each trace
    replays_df = traces_df[traces_df[TIMESTEP_COL] == 0][[TRACE_IDX_COL, TRACE_ID_COL, CLUSTER_ID_COL]]
    file_path = os.path.join(args.output, TRACE_CLUSTERS_FILE)
    replays_df.to_csv(file_path, index=False)

    logging.info('Done!')


if __name__ == '__main__':
    main()
