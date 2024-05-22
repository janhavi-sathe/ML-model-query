env="LaborDivision2"
base="LaborDivision_base"
alg="mahil"

sv="0.0"
dim_c="[2,2]"
exp=""
seed_max=3

echo "env: ${env}, alg: ${alg}, exp: ${exp}, base: ${base}, supervision: ${sv}, dim_c: ${dim_c} max seed: ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train_ma_dnn/run_algs.py alg=${alg} env=${env} \
    base=${base} tag="${exp}Seed${seed}Sv${sv}" supervision=${sv} seed=${seed} dim_c=${dim_c}
done
