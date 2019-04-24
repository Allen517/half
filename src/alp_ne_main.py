from __future__ import print_function

import numpy as np
import random
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from n2v.alp_ne import *
from n2v.utils.graph import Graph

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--graph1', required=True,
                        help='File path of source network')
    parser.add_argument('--graph2', required=True,
                        help='File path of target network')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', required=False,
                        help='If graph format is edgelist, the weighted is functionalized')
    parser.add_argument('--directed', required=False,
                        help='If graph format is edgelist, the weighted is functionalized')
    parser.add_argument('--linkage', required=True,
                        help='Linkage file from source to target network')
    parser.add_argument('--output', required=True,
                        help='Output model file')
    parser.add_argument('--last_emb1', required=False,
                        help='File path of embeddings from source network')
    parser.add_argument('--last_emb2', required=False,
                        help='File path of embeddings from target network')
    parser.add_argument('--log-file', default='ALP_NE',
                        help='logging file')
    parser.add_argument('--lr', default=.01, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=.1, type=float,
                        help='Parameter of regularization')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--rep-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node')
    parser.add_argument('--table-size', default=1e8, type=int,
                        help='Table size')
    parser.add_argument('--saving-step', default=1, type=int,
                        help='The training epochs')
    parser.add_argument('--max-epochs', default=20, type=int,
                        help='The training epochs')
    parser.add_argument('--method', required=True, choices=['alp-ne'],
                        help='The applied methods')
    parser.add_argument('--neg-ratio', default=5, type=int,
                        help='The negative ratio of ALP_NE')
    args = parser.parse_args()
    return args

def main(args):
    SAVING_STEP = args.saving_step
    MAX_EPOCHS = args.max_epochs

    print("Reading...")
    graphs = defaultdict(Graph)
    g_path = dict()
    last_embs = dict()
    g_path['f'] = args.graph1
    g_path['g'] = args.graph2
    last_embs['f'] = args.last_emb1
    last_embs['g'] = args.last_emb2
    last_embs = dict()
    for graph_type in ['f', 'g']:
        if args.graph_format == 'adjlist':
            graphs[graph_type].read_adjlist(filename=g_path[graph_type], delimiter=',')
        elif args.graph_format == 'edgelist':
            graphs[graph_type].read_edgelist(filename=g_path[graph_type]\
                                    , weighted=args.weighted, directed=args.directed, delimiter=',')
    if args.method == 'alp-ne':
        model = ALP_NE(graphs, lr=args.lr, gamma=args.gamma, rep_size=args.rep_size, batch_size=args.batch_size
                        , epoch=args.max_epochs, neg_ratio=args.neg_ratio, table_size=args.table_size
                        , anchor_file=args.linkage, last_emb_files=last_embs, saving_epoch=args.saving_step
                        , outfile=args.output
                        , log_file=args.log_file)

if __name__ == "__main__":
    main(parse_args())
