# /bin/bash

p=10
gamma=(1 0.1 0.01 0.001 0.0001 0.00001)
eta=(1 0.1 0.01 0.001 0.0001 0.00001)
lr=(1 0.1 0.01 0.001 0.0001 0.00001)
os=(10 20 30 40 50 60)
nr=(5 10 15 20 25 30)

for k in $(seq 1 1 5)
do
    for l in ${lr[@]}
    do
        for n in ${nr[@]}:
        do
            for g in ${gamma[@]}
            do
                for e in ${eta[@]}
                do
                    for o in ${os[@]}
                    do
                        python src/half_main.py --log-file online-offline.half-dp.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --device :/cpu:0 --gpu-id 0 --feature-src data/online-offline/train-test/feats.online --feature-end data/online-offline/train-test/feats.offline --identity-linkage data/online-offline/train-test/online-offline.douban.anchors.ptrain${p}.train --output res/online-offline.feats.p${p}.lr${l}.negratio${n}.gamma${g}.eta${e}.outsize${o}.times${k} --method half-dp --type-model lin --is-valid True --early-stop True --saving-step 10 --max-epochs 1000 --batch-size 128 --input-size 540 --output-size ${o} --neg-ratio ${n} --lr ${l} --gamma ${g} --eta ${e}
                    done
                done
            done
        done
    done
done