# -*- coding=UTF-8 -*-\n
from eval.eval import Eval_HALF_DP
from eval.measures import *

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter
                            , conflict_handler='resolve')
    parser.add_argument('-feat-src', required=True
                        , help='features from source network')
    parser.add_argument('-feat-end', required=True
                        , help='features from end network')
    parser.add_argument('-linkage', required=True
                        , help='linkage for test')
    parser.add_argument('-model', required=True
                        , help='Model')
    parser.add_argument('-n-model', default=5, type=int
                        , help='Used when the model is trained in integrated manner')
    parser.add_argument('-n-dim', default=15, type=int
                        , help='Number of dimensions from model')
    parser.add_argument('-eval-type', default='mrr'
                        , help='mrr/ca (MRR/Candidate selection)')
    parser.add_argument('-model-type', required=False
                        , help='Model type: [lin/mlp]')
    parser.add_argument('-n-cands', default=9, type=int
                        , help='Number of candidates')
    parser.add_argument('-top-rank', default=3, type=int
                        , help='Top rank of the matching list')
    parser.add_argument('-filter-thres', default=9, type=int
                        , help='Threshold for filtering candidates in each model')
    # parser.add_argument('-col-prop', default=0.8, type=float
    #                     , help='Proportion for collaborative filtering on models')
    parser.add_argument('-output', required=True
                        , help='Output file')

    return parser.parse_args()

def main(args):
    eval_model = Eval_HALF_DP(args.model_type)
    eval_model._init_eval(feat_src=args.feat_src,
                            feat_end=args.feat_end,
                            linkage=args.linkage
                )
    if args.eval_type=='mrr':
        eval_model.calc_mrr_by_dist(model=args.model, n_model=args.n_model
                                    , candidate_num=args.n_cands, dist_calc=hamming_distance
                                    , out_file=args.output)
    if args.eval_type=='ca':
        dim_index, model_res = eval_model.build_index(model=args.model, n_dim=args.n_dim
                                                    , n_model=args.n_model)
        eval_model.choose_candidates(dim_index=dim_index, model_res=model_res
                                    , top_rank=args.top_rank, out_file=args.output)
    
if __name__=='__main__':
    main(parse_args())



