#!/bin/sh


experiment_name='run_ablation_ff_optimizer'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi

networks=('resnet18')
datasets=('cifar10')
is_finetunes=(1)
label_mapping_modes=('flm')
prune_methods=('imp')
prompt_methods=('None')
input_sizes=(128)
pad_sizes=(48)
pruning_times=1
epochs=200
seed=7

optimizers=('sgd' 'adam')
lr_schedulers=('multistep' 'cosine')
gpus=(7 6 5 4)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for m in ${!is_finetunes[@]};do
            for n in ${!prompt_methods[@]};do
                for l in ${!prune_methods[@]};do
                    for o in ${!optimizers[@]};do
                        for p in ${!lr_schedulers[@]};do
                            log_filename=${foler_name}/ff_optimizer_${optimizers[o]}_${lr_schedulers[p]}.log
                            python ./experiments/main.py \
                                --experiment_name ${experiment_name} \
                                --dataset ${datasets[i]} \
                                --network ${networks[j]} \
                                --label_mapping_mode ${label_mapping_modes[0]} \
                                --prune_method ${prune_methods[l]} \
                                --is_finetune ${is_finetunes[m]} \
                                --prompt_method ${prompt_methods[n]} \
                                --optimizer ${optimizers[o]} \
                                --lr_scheduler ${lr_schedulers[p]} \
                                --gpu ${gpus[2*o+p]} \
                                --input_size ${input_sizes[0]} \
                                --pad_size ${pad_sizes[0]} \
                                --pruning_times ${pruning_times} \
                                --epochs ${epochs} \
                                --seed ${seed} \
                                > $log_filename 2>&1 &
                        done
                    done
                    wait
                done
                wait
            done
            wait
        done
        wait
    done
    wait
done
