#!/bin/sh

experiment_name='ablation_network_train_exp'
if [ ! -d ${experiment_name} ]; then
    mkdir -p ${experiment_name}
fi

prune_methods=('imp')
label_mapping_modes=('ilm')
prompt_methods=('pad')
gpus=(0)
input_sizes=(160)
pad_sizes=(32)
pruning_times=3
epochs=3
seed=7
log_filename=${experiment_name}/network_train_false.log
python ./experiments/prompt_sparsity/lm_vp_prune.py \
    --experiment_name ${experiment_name} \
    --prune_method ${prune_methods[0]} \
    --label_mapping_mode ${label_mapping_modes[0]} \
    --prompt_method ${prompt_methods[0]} \
    --gpu ${gpus[0]} \
    --input_size ${input_sizes[0]} \
    --pad_size ${pad_sizes[0]} \
    --pruning_times ${pruning_times} \
    --epochs ${epochs} \
    --seed ${seed} \
    > $log_filename 2>&1 &
