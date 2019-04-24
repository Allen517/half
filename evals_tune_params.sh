# /bin/bash

thres=(8 9 10)
prop=(0.6 0.7 0.8)

for t in ${thres[@]}
do
    for p in ${prop[@]}
    do
        python src/evals_half.py -feat-src data/online-offline/train-test/feats.online data/online-offline/train-test/feats.alpne.p10.neg5.lr.1.gamma.1.s16.online -feat-end data/online-offline/train-test/feats.offline data/online-offline/train-test/feats.alpne.p10.neg5.lr.1.gamma.1.s16.offline -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -models res/online-offline/inputs_feat/online-offline.feats.p10.lr0.1.negratio10.gamma.1.eta.01.outsize15 res/online-offline/inputs_alpne_s16/online-offline.alpne.s16.p10.lr1.negratio5.gamma1.eta.01.outsize15 -eval-type ca -n-models 5 5 -n-dims 15 15 -model-type lin -output cands.half.integrate.alpne_s16.p10.thres${t}.prop${p} -filter-thres $t -col-prop $p
    done
done