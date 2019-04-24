from __future__ import print_function

from half.half_sp import *
from half.half_dp import *

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
import numpy as np
import random
import time

from utils.LogHandler import LogHandler

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--feature-src', required=False,
                        help='Input embeddings of source network')
    parser.add_argument('--feature-end', required=False,
                        help='Input embeddings of end network')
    parser.add_argument('--type-model', default='mlp', type=str,
                        help='Model type [mlp/lin]')
    parser.add_argument('--identity-linkage', required=False,
                        help='Input linkage from source to end networks')
    parser.add_argument('--output', required=True,
                        help='Output model file')
    parser.add_argument('--log-file', default='HALF',
                        help='logging file')
    parser.add_argument('--lr', default=.01, type=float,
                        help='Learning rate')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Gamma (parameter for binarization)')
    parser.add_argument('--eta', default=0.01, type=float,
                        help='Eta (parameter for code balance)')
    parser.add_argument('--is-valid', default=False, type=bool,
                        help='If use validation in training')
    parser.add_argument('--is-train', default=True, type=bool,
                        help='If the model is used for training or test')
    parser.add_argument('--neg-ratio', default=5, type=int,
                        help='The negative ratio')
    parser.add_argument('--input-size', default=256, type=int,
                        help='Number of embedding')
    parser.add_argument('--hidden-size', default=32, type=int,
                        help='Number of embedding (If needed)')
    parser.add_argument('--output-size', default=32, type=int,
                        help='Number of output code')
    parser.add_argument('--layers', default=2, type=int,
                        help='Number of layers (If choose MLP model)')
    parser.add_argument('--saving-step', default=1, type=int,
                        help='The training epochs')
    parser.add_argument('--early-stop', default=False, type=bool,
                        help='Early stop')
    parser.add_argument('--max-epochs', default=21, type=int,
                        help='The training epochs')
    parser.add_argument('--method', required=True, choices=['half-sp', 'half-dp'],
                        help='The learning methods')
    parser.add_argument('--device', default=':/gpu:0',
                        help='Running device')
    parser.add_argument('--gpu-id', required=False,
                        help='Set env CUDA_VISIBLE_DEVICES', default="0")
    args = parser.parse_args()
    return args

def main(args):
    t1 = time.time()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    logger = LogHandler('RUN.'+time.strftime('%Y-%m-%d',time.localtime(time.time())))
    logger.info(args)

    SAVING_STEP = args.saving_step
    MAX_EPOCHS = args.max_epochs
    files = {
                'feat-src':args.feature_src, 
                'feat-end':args.feature_end, 
                'linkage':args.identity_linkage
            }
    if args.method == 'half-sp':
        model = HALF_SP(learning_rate=args.lr, batch_size=args.batch_size
                        , neg_ratio=args.neg_ratio, gamma=args.gamma, eta=args.eta
                        , n_input=args.input_size, n_out=args.output_size, n_hidden=args.hidden_size
                        , n_layer=args.layers, is_valid=args.is_valid
                        , files=files
                        , type_model=args.type_model
                        , log_file=args.log_file, device=args.device)
    if args.method == 'half-dp':
        model = HALF_DP(learning_rate=args.lr, batch_size=args.batch_size
                        , neg_ratio=args.neg_ratio, gamma=args.gamma, eta=args.eta
                        , n_input=args.input_size, n_out=args.output_size, n_hidden=args.hidden_size
                        , n_layer=args.layers, is_valid=args.is_valid
                        , files=files
                        , type_model=args.type_model
                        , log_file=args.log_file, device=args.device)

    losses = np.zeros(MAX_EPOCHS)
    val_scrs = np.zeros(MAX_EPOCHS)
    best_scr = .0
    best_epoch = 0
    thres = 3
    for i in range(1,MAX_EPOCHS+1):
        losses[i-1], val_scrs[i-1] = model.train_one_epoch()
        if i%SAVING_STEP==0:
            loss_mean = np.mean(losses[i-SAVING_STEP:i])
            scr_mean = np.mean(val_scrs[i-SAVING_STEP:i])
            logger.info('loss in last {} epoches: {}, validation in last {} epoches: {}'
                .format(SAVING_STEP, loss_mean, SAVING_STEP, scr_mean))
            if scr_mean>best_scr:
                best_scr = scr_mean
                best_epoch = i
                model.save_models(args.output)
            if args.early_stop and i>=thres*SAVING_STEP:
                cnt = 0
                for k in range(thres-1,-1,-1):
                    cur_val = np.mean(val_scrs[i-(k+1)*SAVING_STEP:i-k*SAVING_STEP])
                    if cur_val<=best_scr:
                        cnt += 1
                if cnt==thres and (i-best_epoch)>=thres*SAVING_STEP:
                    logger.info('*********early stop*********')
                    logger.info('The best epoch: {}\nThe validation score: {}'.format(best_epoch, best_scr))
                    break
    t2 = time.time()
    print('time cost:',t2-t1)

if __name__ == "__main__":
    main(parse_args())
