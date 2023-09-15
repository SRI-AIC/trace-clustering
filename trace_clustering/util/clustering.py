import copy
import logging
import queue
import numpy as np
import pandas as pd
import itertools as it
import plotly.express as px
import plotly.graph_objs as go
import tqdm
from enum import IntEnum
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from .io import get_file_changed_extension
from .plot import plot_timeseries, format_and_save_plot, DEF_PALETTE, DEF_TEMPLATE

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

SILHOUETTE_COEFFICIENT = 'Silhouette Coefficient'
CALINSKI_HARABASZ_INDEX = 'Calinski-Harabasz Index'
DAVIES_BOULDIN_INDEX = 'Davies-Bouldin Index'

ADJUSTED_RAND_INDEX = 'Adjusted Rand Index'
MUTUAL_INFO_SCORE = 'Adjusted Mutual Information'
HOMOGENEITY_SCORE = 'Homogeneity'
COMPLETENESS_SCORE = 'Completeness'
V_MEASURE_SCORE = 'V-measure'
FOWLKES_MALLOWS_INDEX = 'Fowlkes-Mallows index'


class Orientation(IntEnum):
    """
    Represents possible dendrogram plot orientations.
    """
    top = 1
    right = 2
    bottom = 3
    left = 4


def hopkins_statistic(datapoints: np.ndarray, seed: float = 0) -> float:
    """
    Compute Hopkins statistic [1] for the given datapoints for measuring clustering tendency of data.
    Source: https://github.com/prathmachowksey/Hopkins-Statistic-Clustering-Tendency/blob/master/Hopkins-Statistic-Clustering-Tendency.ipynb
    References:
        - [1] Lawson, R. G., & Jurs, P. C. (1990). New index for clustering tendency and its application to chemical
    problems. Journal of chemical information and computer sciences, 30(1), 36-41.
    https://pubs.acs.org/doi/abs/10.1021/ci00065a010
    :param np.ndarray datapoints: the data for which to compute the Hopkins statistic shaped (n_points, n_features).
    :param float seed: the seed for the RNG used to sample points and generate uniformly distributed points.
    :rtype: float
    :return: the Hopkins's statistic for the given dataset, a value in [0,1] where a value close to 1 tends to indicate
    the data is highly clustered, random data will tend to result in values around 0.5, and uniformly distributed data
    will tend to result in values close to 0.
    """
    rng = np.random.RandomState(seed)
    sample_size = int(datapoints.shape[0] * 0.05)  # 0.05 (5%) based on paper by Lawson and Jures

    # a uniform random sample in the original data space
    simulated_points = rng.uniform(np.min(datapoints, axis=0), np.max(datapoints, axis=0),
                                   (sample_size, datapoints.shape[1]))

    # a random sample of size sample_size from the original data X
    random_indices = rng.choice(np.arange(datapoints.shape[0]), sample_size)
    sample_points = datapoints[random_indices]

    # initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(datapoints)

    # u_distances = nearest neighbour distances from uniform random sample
    u_distances, u_indices = nbrs.kneighbors(simulated_points, n_neighbors=2)
    u_distances = u_distances[:, 0]  # distance to the first (nearest) neighbour

    # w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances, w_indices = nbrs.kneighbors(sample_points, n_neighbors=2)
    # distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[:, 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    # compute and return hopkins' statistic
    return u_sum / (u_sum + w_sum)


def update_clusters(clustering: AgglomerativeClustering, new_distance_threshold: float):
    """
    Updates the cluster labels for each datapoint to be consistent with the algorithm's hierarchy and given distance
    threshold. Useful when we already ran the HAC algorithm to determine the points' hierarchy but want to change the
    threshold at which the number of clusters is found.
    :param AgglomerativeClustering clustering: the clustering algorithm with the distances
    :param float new_distance_threshold: the new distance threshold at which the number of clusters is to be determined.
    :return:
    """
    clustering.distance_threshold = new_distance_threshold
    clustering.labels_ = np.full_like(clustering.labels_, -1, dtype=int)
    _update_clusters(clustering)
    clustering.labels_ = np.max(clustering.labels_) - clustering.labels_  # invert to follow natural order
    clustering.n_clusters_ = int(np.max(clustering.labels_) + 1)


def _update_clusters(clustering: AgglomerativeClustering):
    node_q = queue.Queue()
    node_q.put(len(clustering.children_) - 1)  # work backwards from last node/cluster
    cluster_q = queue.Queue()
    cluster_q.put(0)
    num_clusters = 1
    while not node_q.empty():
        # check to see if we need to split node (if above distance threshold)
        cur_node = node_q.get()
        cur_cluster = cluster_q.get()
        dist = clustering.distances_[cur_node]
        for i, child in enumerate(clustering.children_[cur_node]):
            if i > 0 and dist > clustering.distance_threshold:
                num_clusters += 1
                cur_cluster = num_clusters - 1
            if child < clustering.n_leaves_:
                clustering.labels_[child] = cur_cluster  # child is leaf, assign label
            else:
                node_q.put(child - clustering.n_leaves_)  # child is parent, put in queue
                cluster_q.put(cur_cluster)


def get_sorted_indexes(clustering: AgglomerativeClustering) -> np.ndarray:
    """
    Gets the indexes of the datapoints sorted according to the hierarchy imposed by the given HAC algorithm, i.e.,
    same cluster points will have contiguous indexes and closer points will have a closer index.
    :param AgglomerativeClustering clustering: the clustering result containing the points hierarchy.
    :rtype: np.ndarray
    :return: an array containing the indexes of the datapoints sorted according to the given HAC structure.
    """
    q = queue.Queue()
    q.put(len(clustering.children_) - 1)  # work backwards from last node/cluster
    clusters_idxs = []
    while not q.empty():
        cur_node = q.get()
        for child in clustering.children_[cur_node]:
            if child < clustering.n_leaves_:
                cluster = clustering.labels_[child]
                clusters_idxs.append((cluster, child))  # child is leaf, add to list
            else:
                q.put(child - clustering.n_leaves_)  # child is parent, put in queue

    # groups by cluster, then use clustering order with each cluster
    idxs = []
    for cluster, group in it.groupby(sorted(clusters_idxs), lambda x: x[0]):
        idxs.extend(reversed([idx for cluster, idx in group]))  # closest first
    return np.array(idxs)


def get_n_clusters(clustering: AgglomerativeClustering, n_min: int = 1, n_max: int = 7) -> Dict[int, np.ndarray]:
    """
    Gets clusters (datapoints labels) for a range of different number of clusters given the HAC result.
    :param AgglomerativeClustering clustering: the clustering result containing the hierarchical structure.
    :param int n_min: the minimum number of clusters for which to get the datapoints labels.
    :param int n_max: the maximum number of clusters for which to get the datapoints labels.
    :rtype: Dict[int, np.ndarray]
    :return: an ordered dictionary containing the datapoints' cluster labels for the different number of clusters.
    """
    n_cluster_labels = OrderedDict()
    for n in np.arange(n_min, n_max + 1):
        clustering = copy.copy(clustering)
        if n == clustering.n_leaves_:
            clustering.n_clusters_ = n
            clustering.labels_ = np.arange(n)  # just manually set the labels
        else:
            clustering.distances_ = np.arange(len(clustering.distances_))  # fake distances , keep structure (children)
            dist = _get_distance_num_clusters(clustering, n)
            update_clusters(clustering, dist)
            if clustering.n_clusters_ != n:
                logging.warning(f'Num. resulting clusters {clustering.n_clusters_} != intended: {n}')
        n_cluster_labels[n] = clustering.labels_
    return n_cluster_labels


def _get_distance_num_clusters(clustering: AgglomerativeClustering, n_clusters: int) -> float:
    return clustering.distances_[-1] + 1. if n_clusters <= 1 else \
        clustering.distances_[len(clustering.distances_) - n_clusters]


def get_linkage_matrix(clustering: AgglomerativeClustering) -> pd.DataFrame:
    """
    Gets a linkage matrix from the `sklearn` clustering model.
    See: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    :param AgglomerativeClustering clustering: the clustering model.
    :rtype: np.ndarray
    :return: a linkage matrix containing, for each node (row), the children of the node, the distance at which the node
    was found (children were merged), the number of sub-nodes (leaves) under the node and the label of the cluster
    whose color should be used to draw this node in the dendrogram.
    """
    # create the counts of samples under each node
    counts = np.full(clustering.children_.shape[0], -1, dtype=np.int)
    labels = np.full(clustering.children_.shape[0], -1, dtype=np.int)
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        cur_count = 0
        label = None
        merge = sorted(merge, key=lambda idx: 0. if idx < n_samples else clustering.distances_[idx - n_samples])
        for child_idx in merge:
            if child_idx < n_samples:
                cur_count += 1  # leaf node
                if label is None:
                    label = clustering.labels_[child_idx]
            else:
                cur_count += counts[child_idx - n_samples]
                if label is None:
                    label = labels[int(child_idx - n_samples)]

        labels[i] = label
        counts[i] = cur_count

    return pd.DataFrame({'Child 1': clustering.children_[:, 0],
                         'Child 2': clustering.children_[:, 1],
                         'Distance': clustering.distances_,
                         'Count': counts,
                         'Label': labels},
                        index=pd.RangeIndex(n_samples, n_samples + len(clustering.children_), name='Node'))


def plot_clustering_distances(clustering: AgglomerativeClustering, output_img: str) -> go.Figure:
    """
    Saves a plot with the clustering distances resulting from the given clustering algorithm.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting distances.
    :param str output_img: the path to the file in which to save the plot.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    x_label = 'Num. Clusters'
    y_label = 'Distance'
    title = 'Traces Clustering Distance'

    # creates plot with distances per num. clusters (decreasing to illustrate bottom-up order of HAC)
    num_clusters = np.flip(np.arange(len(clustering.distances_) + 1) + 1)
    distances = np.hstack(([0], clustering.distances_))
    df = pd.DataFrame({x_label: num_clusters, y_label: distances})
    df.set_index(x_label, inplace=True)

    # create line plot
    fig = plot_timeseries(df, title, x_label=x_label, y_label=y_label, reverse_x_axis=True)

    # add line with actual number of clusters
    fig.add_vline(x=clustering.n_clusters_, line_width=2, line_dash='dash', line_color='grey',
                  annotation_text='Selected', annotation_position='top left')

    format_and_save_plot(fig, df, title, output_img, show_legend=False)
    return fig


def plot_clustering_dendrogram(clustering: AgglomerativeClustering,
                               title: Optional[str] = None,
                               output_img: Optional[str] = None,
                               save_csv: bool = True,
                               save_json: bool = True,
                               labels: Optional[List[str]] = None,
                               orientation: Orientation = Orientation.bottom,
                               palette: Union[List[str], str] = DEF_PALETTE,
                               template: str = DEF_TEMPLATE,
                               width: Optional[int] = 800,
                               height: Optional[int] = 600,
                               show_plot: bool = False) -> go.Figure:
    """
    Saves a dendrogram plot with the clustering resulting from the given model.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels and distances.
    :param str title: the title of the plot.
    :param str output_img: the path to the file in which to save the plot.
    :param list[str] labels: a list containing a label for each clustering datapoint. If `None`, the cluster index of
    each datapoint is used as label.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param orientation orientation: the plots orientation ('top', 'right', 'bottom', or 'left').
    :param str or list[str] palette: the name of the Plotly palette used to color each series line, or a list containing
    a string representation for the colors to be used. See: https://plotly.com/python/builtin-colorscales/.
    :param str template: the name of the Plotly layout template to be used. Defaults to "plotly_white".
    See: https://plotly.com/python/templates/.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    # saves linkage info to csv
    linkage_df = get_linkage_matrix(clustering)
    if save_csv and output_img is not None:
        linkage_df.to_csv(get_file_changed_extension(output_img, 'csv'))

    # check palette, get amount of colors matching num. clusters
    if palette is None:
        palette = DEF_PALETTE
    if isinstance(palette, str):
        colors = px.colors.sample_colorscale(palette, np.linspace(0, 1, clustering.n_clusters_))
    else:
        colors = [palette[i % len(palette)] for i in range(clustering.n_clusters_)]

    def _get_node_color(node_idx):
        # get color from label associated with dendrogram node
        if node_idx < clustering.n_leaves_:
            cluster_label = clustering.labels_[node_idx]
        else:
            cluster_label = int(linkage_df.loc[node_idx]['Label'])
        return colors[cluster_label]

    # creates dendrogram (tree) structure
    linkage_matrix = linkage_df.values[:, :4].astype('float64')  # ignore labels column
    d = dendrogram(linkage_matrix, p=clustering.n_clusters_, truncate_mode='level', color_threshold=None,
                   labels=labels, no_plot=True, link_color_func=_get_node_color)

    # based on plotly.figure_factory._dendrogram._Dendrogram
    xaxis, yaxis = 'xaxis', 'yaxis'
    sign = {xaxis: 1 if orientation in [Orientation.left, Orientation.bottom] else -1,
            yaxis: 1 if orientation in [Orientation.right, Orientation.bottom] else -1}

    # based on plotly.figure_factory._dendrogram._Dendrogram.get_dendrogram_traces
    icoord = np.array(d['icoord'])
    dcoord = np.array(d['dcoord'])
    ordered_labels = d['ivl']
    color_list = d['color_list']

    # gets correct node indices matching plot order
    def _expand_node(cur_node):
        node_info = linkage_df.loc[cur_node]
        for child_idx in ['Child 2', 'Child 1']:
            child = int(node_info[child_idx])
            if child >= clustering.n_leaves_:
                d_nodes.append(child)
                _expand_node(child)

    d_nodes = [linkage_df.index[-1]]
    _expand_node(d_nodes[0])
    d_nodes.reverse()

    # create empty figure and then add traces for each tree branch
    hover_template = '%{text}<br>Distance: %{y:.2f}'
    fig = go.Figure(layout=dict(template=template, title=title))
    for i in range(len(icoord)):
        # xs and ys are arrays of 4 points that make up the '∩' shapes of the dendrogram tree
        xs = icoord[i] if orientation in [Orientation.top, Orientation.bottom] else dcoord[i]
        ys = dcoord[i] if orientation in [Orientation.top, Orientation.bottom] else icoord[i]

        # get hover information for each point forming the '∩' trace
        node_info = linkage_df.loc[d_nodes[i]]
        child1 = int(node_info['Child 1'])
        child2 = int(node_info['Child 2'])
        child1_count = 1 if child1 < clustering.n_leaves_ else int(linkage_df.loc[child1]['Count'])
        child2_count = 1 if child2 < clustering.n_leaves_ else int(linkage_df.loc[child2]['Count'])
        child1_label = clustering.labels_[child1] if child1 < clustering.n_leaves_ else \
            int(linkage_df.loc[child1]['Label'])
        child2_label = clustering.labels_[child2] if child2 < clustering.n_leaves_ else \
            int(linkage_df.loc[child2]['Label'])
        text = [f'Count: {child1_count}<br>Label: {child1_label}',  # bottom left
                f'Count: {child1_count}<br>Label: {child1_label}',  # top left
                f'Count: {child2_count}<br>Label: {child2_label}',  # top right
                f'Count: {child2_count}<br>Label: {child2_label}']  # bottom right

        # add '∩' trace to figure
        fig.add_scatter(
            x=np.multiply(sign[xaxis], xs),
            y=np.multiply(sign[yaxis], ys),
            name='',
            mode='lines',
            marker=dict(color=color_list[i]),
            hovertemplate=hover_template,
            text=text
        )

    # add distance (threshold) line to plot
    dist_thresh = clustering.distances_[len(clustering.distances_) - clustering.n_clusters_ + 1] \
        if clustering.distance_threshold is None else clustering.distance_threshold
    if orientation in [Orientation.top, Orientation.bottom]:
        fig.add_hline(y=dist_thresh, line_width=2, line_dash='dash', line_color='grey',
                      annotation_text='Distance threshold', annotation_position='bottom right')
    else:
        fig.add_vline(x=dist_thresh, line_width=2, line_dash='dash', line_color='grey',
                      annotation_text='Distance threshold', annotation_position='top right')

    yvals_flat = dcoord.flatten()
    xvals_flat = icoord.flatten()
    zero_vals = []
    for i in range(len(yvals_flat)):
        if yvals_flat[i] == 0.0 and xvals_flat[i] not in zero_vals:
            zero_vals.append(xvals_flat[i])

    if len(zero_vals) > len(dcoord) + 1:
        # If the length of zero_vals is larger than the length of yvals, it means that there are wrong vals because
        # of the identical samples. Three and more identical samples will make the yvals of splitting center into 0
        # and it will accidentally take it as leaves.
        l_border = int(min(zero_vals))
        r_border = int(max(zero_vals))
        correct_leaves_pos = range(l_border, r_border + 1, int((r_border - l_border) / len(dcoord)))
        # Regenerating the leaves pos from the self.zero_vals with equally intervals.
        zero_vals = [v for v in correct_leaves_pos]
    zero_vals.sort()

    # based on plotly.figure_factory._dendrogram._Dendrogram.set_axis_layout
    axis_layout = dict(type='linear', ticks='outside', rangemode='tozero',
                       showticklabels=True, zeroline=False, showline=True)
    fig.layout[xaxis] = axis_layout
    fig.layout[yaxis] = dict(axis_layout)

    if len(ordered_labels) > 0:
        axis_key = yaxis if orientation in [Orientation.left, Orientation.right] else xaxis
        fig.layout[axis_key].update(dict(
            tickvals=[zv * sign[axis_key] for zv in zero_vals], ticktext=ordered_labels, tickmode='array'),
            range=[0, max(zero_vals) + min(zero_vals)])

    axis_key = xaxis if orientation in [Orientation.left, Orientation.right] else yaxis
    fig.layout[axis_key].update(dict(title='Distance'))

    # format and save plot
    fig.update_layout(dict(hovermode='closest'))
    format_and_save_plot(
        fig, None, title, output_img, save_csv=False, save_json=save_json,
        width=width, height=height, show_legend=False, show_plot=show_plot)

    return fig


def get_internal_evaluations(clustering: AgglomerativeClustering,
                             max_clusters: int,
                             data: Optional[np.ndarray],
                             dist_matrix: Optional[np.ndarray]) -> Dict[str, Dict[int, float]]:
    """
    Performs internal clustering evaluation using different metrics for different number of clusters.
    :param AgglomerativeClustering clustering: the hierarchical clustering model to be evaluated.
    :param int max_clusters: the maximum number of clusters for which to perform evaluation.
    :param np.ndarray data: the data that was used to produce the clustering model.
    :param np.ndarray dist_matrix: a matrix containing the pairwise distances between all the datapoints.
    :return:
    """
    if data is None and dist_matrix is None:
        raise ValueError(f'Need to provide one or both of "data" and "dist_matrix"!')

    max_clusters = min(max(clustering.n_clusters_, max_clusters), clustering.n_leaves_ - 1)
    n_clusters = get_n_clusters(clustering, 2, max_clusters)  # gets all clustering partitions for diff num. clusters
    evals: Dict[str, Dict[int, float]] = {
        SILHOUETTE_COEFFICIENT: OrderedDict(),
        CALINSKI_HARABASZ_INDEX: OrderedDict(),
        DAVIES_BOULDIN_INDEX: OrderedDict()
    }
    for n, labels in tqdm.tqdm(n_clusters.items()):
        if dist_matrix is not None:
            evals[SILHOUETTE_COEFFICIENT][n] = metrics.silhouette_score(dist_matrix, labels, metric='precomputed')
        if data is not None:
            if dist_matrix is None:
                evals[SILHOUETTE_COEFFICIENT][n] = metrics.silhouette_score(data, labels)
            evals[CALINSKI_HARABASZ_INDEX][n] = metrics.calinski_harabasz_score(data, labels)
            evals[DAVIES_BOULDIN_INDEX][n] = metrics.davies_bouldin_score(data, labels)
    return evals


def get_external_evaluations(clustering: AgglomerativeClustering,
                             max_clusters: int,
                             labels_true: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Performs external clustering evaluation using different metrics for different number of clusters.
    :param AgglomerativeClustering clustering: the hierarchical clustering model to be evaluated.
    :param int max_clusters: the maximum number of clusters for which to perform evaluation.
    :param list[int] labels_true: the ground-truth labels for each datapoint.
    :return:
    """
    max_clusters = max(clustering.n_clusters_, max_clusters)
    n_clusters = get_n_clusters(clustering, 2, max_clusters)
    evals = {
        ADJUSTED_RAND_INDEX: OrderedDict(),
        MUTUAL_INFO_SCORE: OrderedDict(),
        HOMOGENEITY_SCORE: OrderedDict(),
        COMPLETENESS_SCORE: OrderedDict(),
        V_MEASURE_SCORE: OrderedDict(),
        FOWLKES_MALLOWS_INDEX: OrderedDict()
    }
    for n, labels_pred in tqdm.tqdm(n_clusters.items()):
        evals[ADJUSTED_RAND_INDEX][n] = metrics.rand_score(labels_true, labels_pred)
        evals[MUTUAL_INFO_SCORE][n] = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        evals[HOMOGENEITY_SCORE][n] = metrics.homogeneity_score(labels_true, labels_pred)
        evals[COMPLETENESS_SCORE][n] = metrics.completeness_score(labels_true, labels_pred)
        evals[V_MEASURE_SCORE][n] = metrics.v_measure_score(labels_true, labels_pred)
        evals[FOWLKES_MALLOWS_INDEX][n] = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    return evals
