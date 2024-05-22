env="Protoss5v5"
base="DiscreteWorld_base"
alg="mahil"

dim_c="[1,1,1,1,1]"
exp="iiql"
seed=1

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, dim_c: ${dim_c} max seed: ${seed_max}"
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
    base=${base} tag="${exp}Seed${seed}" seed=${seed} dim_c=${dim_c}
