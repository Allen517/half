# /bin/bash

# p=10
# g=.1
# e=.01
# l=0.1
# o=15
# n=10

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file online-offline.half-dp.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/online-offline/train-test/feats.online --feature-end data/online-offline/train-test/feats.offline --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 540 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=.1
# e=.01
# l=1
# o=15
# n=10

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file online-offline.half-dp.emb.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/online-offline/online.epoch100.node_order1 --feature-end data/online-offline/offline.epoch100.node_order1 --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.emb.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 16 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=1
# e=1
# l=.1
# o=15
# n=10

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file online-offline.half-dp.cmb.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/online-offline/train-test/feats_cmb.online --feature-end data/online-offline/train-test/feats_cmb.offline --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.cmb.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 556 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# Temp

# p=10
# g=1
# e=0.0001
# l=.1
# o=15
# n=10

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file online-offline.half-dp.cmb.alpne.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/online-offline/train-test/feats_cmb.alpne.online --feature-end data/online-offline/train-test/feats_cmb.alpne.offline --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.cmb.alpne.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 668 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=1
# e=.01
# l=1
# o=15
# n=5

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file online-offline.half-dp.alpne.s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src feats.alpne.p10.neg5.lr.1.gamma.1.s16.online --feature-end feats.alpne.p10.neg5.lr.1.gamma.1.s16.offline --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.alpne.s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 16 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=1
# e=1
# l=.1
# o=15
# n=10

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file online-offline.half-dp.cmb.alpne_s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/online-offline/train-test/feats_cmb.alpne_s16.online --feature-end data/online-offline/train-test/feats_cmb.alpne_s16.offline --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.cmb.alpne_s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 556 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=1
# e=.1
# l=.1
# o=15
# n=15

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file online-offline.half-dp.cmb.alpne_s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/online-offline/train-test/feats_cmb.alpne_s16.online --feature-end data/online-offline/train-test/feats_cmb.alpne_s16.offline --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.cmb.alpne_s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 556 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=.001
# e=.001
# l=0.1
# o=35
# n=20

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file aminer-linkedin.half-dp.name.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/aminer-linkedin/train-test/feats.name.aminer --feature-end data/aminer-linkedin/train-test/feats.name.linkedin --identity-linkage data/aminer-linkedin/train-test/aminer-linkedin.flt_by_feat.anchors.ptrain${p}.train --output res/aminer-linkedin.name.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 26 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=1
# e=.0001
# l=.01
# o=19
# n=25

# for k in $(seq 1 1 5)
# do
#     python src/half_main.py --log-file aminer-linkedin.half-dp.desc.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/aminer-linkedin/train-test/feats.desc.aminer --feature-end data/aminer-linkedin/train-test/feats.desc.linkedin --identity-linkage data/aminer-linkedin/train-test/aminer-linkedin.flt_by_feat.anchors.ptrain${p}.train --output res/aminer-linkedin.desc.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 300 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=1
# e=.01
# l=.01
# o=20
# n=15

# for k in $(seq 1 1 5)
# do
#    python src/half_main.py --log-file aminer-linkedin.half-dp.alp-ne.s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/aminer-linkedin/train-test/feats.alp-ne.s16.aminer --feature-end data/aminer-linkedin/train-test/feats.alp-ne.s16.linkedin --identity-linkage data/aminer-linkedin/train-test/aminer-linkedin.flt_by_feat.anchors.ptrain${p}.train --output res/aminer-linkedin.alp-ne.s16.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 16 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

# p=10
# g=.1
# e=.0001
# l=0.01
# o=16
# n=20

# for k in $(seq 1 1 10)
# do
#     python src/half_main.py --log-file flickr-lastfm.half-dp.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/flickr-lastfm/train-test/feats.flickr --feature-end data/flickr-lastfm/train-test/feats.lastfm --identity-linkage data/flickr-lastfm/train-test/flickr-lastfm.anchors.ptrain${p}.train --output res/flickr-lastfm.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 6 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
# done

p=10
g=1
e=.1
l=.1
o=14
n=10

for k in $(seq 1 1 10)
do
    python src/half_main.py --log-file flickr-myspace.half-dp.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/flickr-myspace/train-test/feats.flickr --feature-end data/flickr-myspace/train-test/feats.myspace --identity-linkage data/flickr-myspace/train-test/flickr-myspace.anchors.ptrain${p}.train --output res/flickr-myspace.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 6 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
done