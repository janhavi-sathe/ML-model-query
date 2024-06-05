env="LaborDivision3-v2"
base="LaborDivision_base"
alg="iiql"

exp=""
seed_max=3

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, max seed: ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
    base=${base} tag="${exp}Seed${seed}" seed=${seed}
done
