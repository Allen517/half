# /bin/bash

# Features

# thres=(7 8 9 10 11)
# prop=(0.5 0.6 0.7 0.8 0.9)

# for t in ${thres[@]}
# do
#   for p in ${prop[@]}
#   do
#       python src/eval_half.py -feat-src data/online-offline/train-test/feats.online -feat-end data/online-offline/train-test/feats.offline -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model res/online-offline.feats.p10.lr0.1.negratio10.gamma.1.eta.01.outsize15 -eval-type ca -n-model 5 -n-dim 15 -model-type lin -output cand.half.p10.thres${t}.prop${p} -filter-thres $t -col-prop $p
#   done
# done

# Embeddings

# thres=(5 6 7)
# prop=(0.5 0.6 0.7 0.8 0.9)

# for t in ${thres[@]}
# do
#     for p in ${prop[@]}
#     do
#         python src/eval_half.py -feat-src data/online-offline/online.epoch100.node_order1 -feat-end data/online-offline/offline.epoch100.node_order1 -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model res/online-offline.emb.p10.lr1.negratio10.gamma.1.eta.01.outsize15 -eval-type ca -n-model 5 -n-dim 15 -model-type lin -output cand.half.p10.thres${t}.prop${p} -filter-thres $t -col-prop $p
#     done
# done

# Combination

# thres=(9 10 11 12 13)
# prop=(0.7 0.8 0.9)

# for t in ${thres[@]}
# do
#     for p in ${prop[@]}
#     do
#         python src/eval_half.py -feat-src data/online-offline/train-test/feats_cmb.online -feat-end data/online-offline/train-test/feats_cmb.offline -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model res/online-offline.cmb.p10.lr.1.negratio10.gamma1.eta1.outsize15 -eval-type mrr -n-model 5 -n-dim 15 -model-type lin -output mrr.half.p10.thres${t}.prop${p} -filter-thres $t -col-prop $p
#     done
# done

thres=(9 10 11 12 13)
prop=(0.7 0.8 0.9)

for t in ${thres[@]}
do
    for p in ${prop[@]}
    do
        python src/eval_half.py -feat-src data/online-offline/train-test/feats_cmb.alpne.online -feat-end data/online-offline/train-test/feats_cmb.alpne.offline -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model res/online-offline.cmb.alpne.p10.lr.1.negratio10.gamma1.eta0.0001.outsize15 -eval-type mrr -n-model 5 -n-dim 15 -model-type lin -output mrr.half.p10.thres${t}.prop${p} -filter-thres $t -col-prop $p
    done
done