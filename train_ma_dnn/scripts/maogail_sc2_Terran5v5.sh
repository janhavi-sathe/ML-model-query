env="Terran5v5"
base="DiscreteWorld_base"
alg="maogail"

sv="0.0"
dim_c="[3,3,3,3,3]"
exp=""

seed=1
echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, supervision: ${sv}, dim_c: ${dim_c} seed: ${seed}"
CUDA_VISIBLE_DEVICES=0 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
    base=${base} tag="${exp}Seed${seed}Sv${sv}" supervision=${sv} seed=${seed} dim_c=${dim_c}
