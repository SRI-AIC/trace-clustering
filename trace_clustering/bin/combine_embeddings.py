import logging
import os
import argparse
import pandas as pd
import tqdm
from trace_clustering.bin.get_embeddings import EMBEDDINGS_FILE
from trace_clustering.bin.split_data import TRACE_IDX_COL
from trace_clustering.util.io import create_clear_dir
from trace_clustering.util.cmd_line import save_args, str2bool
from trace_clustering.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Loads files containing feature traces (sequences) embeddings and merges them ' \
           'for subsequent trace clustering.'


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, nargs='+',
                        help='Trace embeddings files to be concatenated/merged.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save results')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    if args.input is None or len(args.input) == 0:
        raise ValueError(f'Invalid input provided: {args.input}!')
    create_clear_dir(args.output, args.clear)
    save_args(args, os.path.join(args.output, 'args.json'))
    change_log_handler(os.path.join(args.output, 'get-embeddings.log'), args.verbosity)

    logging.info('========================================')
    logging.info(f'Checking {len(args.input)} input files...')
    embeddings_dfs = []
    for i, embed_file in tqdm.tqdm(enumerate(args.input), total=len(args.input)):
        # check and load embeddings file
        if not os.path.isfile(embed_file):
            logging.info(f'Nonexistent feature file "{embed_file}", skipping...')
            continue
        df = pd.read_pickle(embed_file)
        df.set_index(TRACE_IDX_COL, inplace=True)
        logging.info(f'Got {df.shape[0]} trace embeddings for {df.shape[1]} embedding features')
        embeddings_dfs.append(df)

    # merge embeddings
    embeddings_df = pd.concat(embeddings_dfs, axis=1, join='inner')
    embeddings_df.rename(columns={'index': TRACE_IDX_COL})
    embeddings_df.sort_values(TRACE_IDX_COL, inplace=True, ascending=True)
    embeddings_df.reset_index(drop=True, inplace=True)
    logging.info(f'Total of {embeddings_df.shape[0]} trace embeddings '
                 f'for {embeddings_df.shape[1] - 1} embedding features')

    # save merged embeddings to compressed file
    logging.info('========================================')
    file_path = os.path.join(args.output, EMBEDDINGS_FILE)
    logging.info(f'Saving CSV file with all traces embeddings to: {file_path}...')
    embeddings_df.to_pickle(file_path, compression='gzip')

    logging.info('Done!')


if __name__ == '__main__':
    main()
