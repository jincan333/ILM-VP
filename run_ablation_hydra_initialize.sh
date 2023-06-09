#!/bin/sh

experiment_name='ablation_hydra_init_exp'
if [ ! -d ${experiment_name} ]; then
    mkdir -p ${experiment_name}
fi


prune_methods=('hydra')
label_mapping_modes=('ilm')
prompt_methods=('pad')
hydra_scaled_inits=(1 0)
gpus=(7 6)
input_sizes=(128)
pad_sizes=(48)
mask_sizes=(183)
pruning_times=10
epochs=200
seed=7
for i in ${!hydra_scaled_inits[@]};do
    log_filename=${experiment_name}/hydra_initialize_${hydra_scaled_inits[i]}.log
    python ./experiments/prompt_sparsity/lm_vp_prune.py \
        --experiment_name ${experiment_name} \
        --prune_method ${prune_methods[0]} \
        --label_mapping_mode ${label_mapping_modes[0]} \
        --prompt_method ${prompt_methods[0]} \
        --hydra_scaled_init ${hydra_scaled_inits[i]} \
        --gpu ${gpus[i]} \
        --input_size ${input_sizes[0]} \
        --pad_size ${pad_sizes[0]} \
        --mask_size ${mask_sizes[0]} \
        --pruning_times ${pruning_times} \
        --epochs ${epochs} \
        --seed ${seed} \
        > $log_filename 2>&1 &
done