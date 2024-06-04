env="Protoss5v5"
base="DiscreteWorld_base"
alg="mahil"

sv="0.0"
dim_c="[3,3,3,3,3]"
exp=""
seed=3

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, supervision: ${sv}, dim_c: ${dim_c} seed: ${seed}"
CUDA_VISIBLE_DEVICES=1 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
    base=${base} tag="${exp}Seed${seed}Sv${sv}" supervision=${sv} seed=${seed} dim_c=${dim_c}
