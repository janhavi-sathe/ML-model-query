env="Terran5v5"
base="DiscreteWorld_base"
alg="magail"

exp=""
seed=1

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, seed: ${seed}"
CUDA_VISIBLE_DEVICES=0 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
base=${base} tag="${exp}Seed${seed}" seed=${seed}
