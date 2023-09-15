import logging
import os
import argparse
import pandas as pd
import numpy as np
import tqdm
from collections import OrderedDict
from trace_clustering.util.cmd_line import save_args, str2bool
from trace_clustering.util.data import save_separate_csv_gzip
from trace_clustering.util.io import create_clear_dir
from trace_clustering.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Loads a series of datasets containing feature traces (sequences) and splits the data into' \
           'training (clustering) and test sets. Ground-truth partitions can be specified via cmd-line.'

# defaults based on SC2 traces using feature_extractor
DEF_FEATURES_START_IDX = 2
DEF_TRACE_ID_COL = 'File'
DEF_TIMESTEP_COL = 'Timestep'

# new trace dataset columns
TRACE_IDX_COL = 'Trace index'
TIMESTEP_COL = 'Timestep'
TRACE_ID_COL = 'Trace ID'
PARTITION_COL = 'Partition'
FEATURES_START_IDX = 4  # final traces file will contain trace index, timestep, trace id and partition as metadata cols
TRACES_FILE = 'all-traces'


def _select_and_save_traces(traces_df: pd.DataFrame, trace_ids: np.ndarray, file_path: str, separate_episodes: bool):
    # select, re-index, and sort traces
    traces_df = traces_df[traces_df[TRACE_ID_COL].isin(trace_ids)]
    traces_df = traces_df.copy()  # make copy to keep original df intact
    for i, (g, _) in enumerate(traces_df.groupby(TRACE_ID_COL)):
        traces_df.loc[traces_df[TRACE_ID_COL] == g, TRACE_IDX_COL] = i
    traces_df.sort_values([TRACE_IDX_COL, TIMESTEP_COL], inplace=True, ascending=[True, True])

    if separate_episodes:
        # saves separate files for each episode in one gzip file
        save_separate_csv_gzip(traces_df, file_path, group_by=TRACE_ID_COL, use_group_filename=True)
    else:
        # save to single compressed csv file
        logging.info(f'Saving compressed file with all traces data to: {file_path}...')
        traces_df.to_pickle(file_path, compression='gzip')

    return traces_df


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, nargs='+',
                        help='Dataset files containing high-level feature sequences (traces).')
    parser.add_argument('--partitions', '-p', type=str, default=None, nargs='+',
                        help='Ground-truth partition labels for each file provided in input.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save results')
    parser.add_argument('--seed', type=int, default=0, help='Seed used for random number generation.')
    parser.add_argument('--split', '-s', type=float, default=1,
                        help='The ratio (value in (0,1]) of input data to be used for clustering. '
                             'The rest of the trace data will be saved in a separate file.')

    parser.add_argument('--trace-col', type=str, default=DEF_TRACE_ID_COL,
                        help='Column specifying the trace unique identifier (file name, index, etc.)')
    parser.add_argument('--timestep-col', type=str, default=DEF_TIMESTEP_COL,
                        help='Column specifying the timestep index within each trace/episode')
    parser.add_argument('--start-col', type=int, default=DEF_FEATURES_START_IDX,
                        help='Index of first column corresponding to a feature, i.e., not metadata. '
                             'It is assumed subsequent columns are also feature columns')

    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    if not (0 < args.split <= 1):
        raise ValueError(f'Data split must be a number in (0, 1]: {args.split}!')

    if args.input is None or len(args.input) == 0:
        raise ValueError(f'Invalid input provided: {args.input}!')
    create_clear_dir(args.output, args.clear)
    save_args(args, os.path.join(args.output, 'args.json'))
    change_log_handler(os.path.join(args.output, 'split_data.log'), args.verbosity)

    # gets input files
    logging.info('========================================')
    logging.info(f'Checking {len(args.input)} input files...')
    metadata_cols = OrderedDict({TRACE_IDX_COL: None, args.timestep_col: None,
                                 args.trace_col: None, PARTITION_COL: None})  # set this col order
    feature_cols = OrderedDict({})
    traces_dfs = []
    total_traces = 0
    for i, feat_file in tqdm.tqdm(enumerate(args.input), total=len(args.input)):
        # check and load features file
        if not os.path.isfile(feat_file):
            logging.info(f'Nonexistent feature file "{feat_file}", skipping...')
            continue

        partition = args.partitions[i] if args.partitions is not None and len(args.partitions) > i else f'Partition {i}'
        logging.info(f'Loading file: {feat_file} (partition: {partition})...')
        if feat_file.endswith('.pkl') or feat_file.endswith('.pkl.gz'):
            df = pd.read_pickle(feat_file)
        elif feat_file.endswith('.csv') or feat_file.endswith('.csv.gz'):
            df = pd.read_csv(feat_file)
        else:
            logging.info(f'Unknown feature file type: "{feat_file}", skipping...')
            continue

        # check columns
        if args.trace_col not in df.columns or args.timestep_col not in df.columns:
            logging.info(f'Trace ID or timesteps column missing from feature file "{feat_file}", skipping...')
            continue
        logging.info(f'Processing feature file "{feat_file}"...')
        feature_cols.update({col: None for col in df.columns[args.start_col:]})  # update feature columns list

        # set trace indexes
        unique_trace_ids = df[args.trace_col].unique()
        logging.info(f'Found {len(unique_trace_ids)} unique traces, setting indexes...')
        for j, trace_id in enumerate(unique_trace_ids):
            df.loc[df[args.trace_col] == trace_id, TRACE_IDX_COL] = total_traces + j  # set trace index
        df[PARTITION_COL] = partition  # add partition column
        traces_dfs.append(df)  # append

        total_traces += len(unique_trace_ids)
        logging.info(f'Added {len(df)} timesteps')

    if len(traces_dfs) == 0:
        raise ValueError(f'No valid input files provided: {args.input}!')

    logging.info(f'\nFinished. Loaded a total of {total_traces} traces from {len(args.input)} files')
    traces_df = pd.concat(traces_dfs, ignore_index=True)
    traces_df[TRACE_IDX_COL] = traces_df[TRACE_IDX_COL].astype(int)
    traces_df = traces_df[list(metadata_cols) + list(feature_cols)]  # select metadata and feature cols only
    traces_df.rename(columns={args.trace_col: TRACE_ID_COL,
                              args.timestep_col: TIMESTEP_COL}, inplace=True)  # rename cols for subsequent tasks
    traces_df.sort_values([TRACE_IDX_COL, TIMESTEP_COL], inplace=True, ascending=[True, True])  # sanity check step

    # sort and split traces into train + test sets
    logging.info('\n========================================')
    logging.info(f'Splitting clustering (train) and test data based on a {args.split}-{1 - args.split} ratio')
    trace_ids = traces_df[TRACE_ID_COL].unique()
    np.random.RandomState(args.seed).shuffle(trace_ids)
    num_train = int(args.split * len(trace_ids))
    num_test = len(trace_ids) - num_train

    # save traces to compressed files
    logging.info(f'Saving test set ({num_test} traces)...')
    if num_test > 0:
        _select_and_save_traces(traces_df, trace_ids[num_train:],
                                os.path.join(args.output, TRACES_FILE + '-test.tar.gz'), separate_episodes=True)

    logging.info(f'Saving training set ({num_train} traces)...')
    _select_and_save_traces(traces_df, trace_ids[:num_train],
                            os.path.join(args.output, TRACES_FILE + '-train.pkl.gz'), separate_episodes=False)

    logging.info('Done!')


if __name__ == '__main__':
    main()
