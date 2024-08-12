env="Protoss5v5"
base=""
alg="bc"

exp=""
seed_max=3

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, max seed: ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
    base=${base} tag="${exp}Seed${seed}" seed=${seed}
done
