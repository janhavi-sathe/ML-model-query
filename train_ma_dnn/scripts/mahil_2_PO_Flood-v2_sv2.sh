env="PO_Flood-v2"
base="DiscreteWorld_base"
alg="mahil"

sv="0.2"
dim_c="[4,4]"
exp="short"
seed_max=3

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, supervision: ${sv}, dim_c: ${dim_c} max seed: ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
    base=${base} tag="${exp}Seed${seed}Sv${sv}" supervision=${sv} seed=${seed} dim_c=${dim_c}
done
