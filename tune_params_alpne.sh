NET='/home/wangyongqing/exps/dcn/DCNH/data/blogCatalog/inb_dat'
LABEL='/home/wangyongqing/exps/dcn/DCNH/data/blogCatalog/inb_dat/labels'
RES='alp_ne_res/bc.alp_ne'

for t in $(seq .4 .2 .8)
do
    for p in $(seq .1 .2 .9)
    do
        python src/alp_ne_main.py --method alp-ne --graph1 data/online-offline/online.adjlist --graph2 data/online-offline/offline.adjlist --log-file online-offline.alp-ne.p10 --batch-size 256 --table-size 1000000 --saving-step 10 --max-epochs 100 --neg-ratio 5 --linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain10.train --output online-offline.alp-ne.p10.neg5.lr.1.gamma.1 --lr 0.1 --gamma 0.1
    done
done