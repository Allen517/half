python src/eval_half.py -feat-src data/online-offline/train-test/feats.online -feat-end data/online-offline/train-test/feats.offline -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model res/online-offline.feats.p10 -model-type lin -output mrr.half.p10


python src/eval_half.py -feat-src data/online-offline/train-test/feats.online -feat-end data/online-offline/train-test/feats.offline -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model res/online-offline.feats.p10.lr0.1.negratio10.gamma.1.eta.01.outsize15 -n-model 5 -model-type lin -output mrr.half.integrate.p10


python src/eval_half.py -feat-src data/online-offline/train-test/feats.online -feat-end data/online-offline/train-test/feats.offline -linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model res/online-offline.feats.p10.lr0.1.negratio10.gamma.1.eta.01.outsize15 -n-model 5 -n-dim 15 -model-type lin -output cand.half.integrate.p10 -filter-thres 8 -col-prop 1.