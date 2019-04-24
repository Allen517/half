# -*- coding=UTF-8 -*-\n
from eval.evals import Evals_HALF_DP
from eval.measures import *

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter
                            , conflict_handler='resolve')
    parser.add_argument('-feat-src', nargs='+', required=True
                        , help='features from source network')
    parser.add_argument('-feat-end', nargs='+', required=True
                        , help='features from end network')
    parser.add_argument('-linkage', required=True
                        , help='linkage for test')
    parser.add_argument('-models', nargs='+', required=True
                        , help='Model')
    parser.add_argument('-n-models', nargs='+', required=True, type=int
                        , help='Used when the model is trained in integrated manner')
    parser.add_argument('-eval-type', default='mrr'
                        , help='mrr/ca (MRR/Candidate selection)')
    parser.add_argument('-model-type', required=False
                        , help='Model type: [lin/mlp]')
    parser.add_argument('-n-cands', default=9, type=int
                        , help='Number of candidates')
    parser.add_argument('-n-dims', nargs='+', required=True, type=int
                        , help='Number of dimensions from model (for ca)')
    parser.add_argument('-filter-thres', default=9, type=int
                        , help='Threshold for filtering candidates in each model (for ca)')
    parser.add_argument('-col-prop', default=0.8, type=float
                        , help='Proportion for collaborative filtering on models (for ca)')
    parser.add_argument('-output', required=True
                        , help='Output file')

    return parser.parse_args()

def main(args):
    eval_model = Evals_HALF_DP(args.model_type)
    eval_model._init_eval(feat_src=args.feat_src,
                            feat_end=args.feat_end,
                            linkage=args.linkage
                )
    if args.eval_type=='mrr':
        eval_model.calc_mrr_by_dist(models=args.models, n_models=args.n_models
                                    , candidate_num=args.n_cands, dist_calc=hamming_distance
                                    , out_file=args.output)
    if args.eval_type=='ca':
        dim_index, model_res = eval_model.build_index(models=args.models, n_models=args.n_models
                                                    , n_dims=args.n_dims)
        eval_model.choose_candidates(dim_index=dim_index, model_res=model_res
                                    , filter_thres=args.filter_thres, candidate_num=args.n_cands
                                    , out_file=args.output)
    
if __name__=='__main__':
    main(parse_args())



