env="Protoss5v5"
base="DiscreteWorld_base"
alg="magail"

exp=""
seed=3

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, seed: ${seed}"
CUDA_VISIBLE_DEVICES=1 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
base=${base} tag="${exp}Seed${seed}" seed=${seed}
