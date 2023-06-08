#!/bin/sh

# flm_pre&after
experiment_name='ablation_flm_pre&after_exp'
if [ ! -d ${experiment_name} ]; then
    mkdir -p ${experiment_name}
fi

prune_methods=('imp')
label_mapping_modes=('flm')
prompt_methods=('pad')
flm_loc=('pre' 'after')
gpus=(7 6)
input_sizes=(160)
pad_sizes=(32)
pruning_times=3
epochs=3
seed=7
for i in ${!flm_loc[@]};do
    log_filename=${experiment_name}/flm_preafter_${flm_loc[i]}.log
    python ./experiments/prompt_sparsity/lm_vp_prune.py \
        --experiment_name ${experiment_name} \
        --prune_method ${prune_methods[0]} \
        --label_mapping_mode ${label_mapping_modes[0]} \
        --prompt_method ${prompt_methods[0]} \
        --flm_loc ${flm_loc[i]} \
        --gpu ${gpus[i]} \
        --input_size ${input_sizes[0]} \
        --pad_size ${pad_sizes[0]} \
        --pruning_times ${pruning_times} \
        --epochs ${epochs} \
        --seed ${seed} \
        > $log_filename 2>&1 &
done