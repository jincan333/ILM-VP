#!/bin/sh

experiment_name='visual_prompt_finetune'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi


datasets=('cifar10')
networks=('resnet18')
is_finetunes=(1)
label_mapping_modes=('flm')
prune_methods=('imp' 'omp' 'hydra')
prompt_methods=('pad')
optimizers=('sgd')
lr_schedulers=('cosine')
gpus=(7 6 5 4)
input_sizes=(128)
pad_sizes=(48)
pruning_times=10
epochs=200
seed=7
for i in ${!datasets[@]};do
    for j in ${!networks[@]};do
        for l in ${!prune_methods[@]};do
            log_filename=${foler_name}/${datasets[i]}_${networks[j]}_${prune_methods[l]}_${is_finetunes[m]}.log
            python ./experiments/main.py \
                --experiment_name ${experiment_name} \
                --dataset ${datasets[i]} \
                --network ${networks[j]} \
                --label_mapping_mode ${label_mapping_modes[k]} \
                --prune_method ${prune_methods[l]} \
                --is_finetune ${is_finetunes[m]} \
                --prompt_method ${prompt_methods[0]} \
                --optimizer ${optimizers[0]} \
                --lr_scheduler ${lr_schedulers[0]} \
                --gpu ${gpus[l]} \
                --input_size ${input_sizes[0]} \
                --pad_size ${pad_sizes[0]} \
                --pruning_times ${pruning_times} \
                --epochs ${epochs} \
                --seed ${seed} \
                > $log_filename 2>&1 &
        done
        wait
    done
    wait
done
