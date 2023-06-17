#!/bin/sh

experiment_name='ablation_ff_optimizer_scheduler'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# datasets=("cifar100" "dtd" "flowers102" "ucf101" "food101" "gtsrb" "svhn" "eurosat" "oxfordpets" "stanfordcars" "sun397")
# datasets=("ucf101" "eurosat" "oxfordpets" "stanfordcars" "sun397") 
networks=('resnet18')
datasets=('cifar10')
epochs=50
seed=7
prune_modes=('no_tune')
prune_methods=('imp')
label_mapping_modes=('flm')

ff_optimizers=('sgd' 'adam')
ff_schedulers=('multistep' 'cosine')
gpus=(7 4)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for k in ${!prune_modes[@]};do
            for l in ${!prune_methods[@]};do
                for m in ${!label_mapping_modes[@]};do
                    for n in ${!ff_optimizers[@]};do
                        for o in ${!ff_schedulers[@]};do
                            log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_modes[k]}_${prune_methods[l]}_${ff_optimizers[n]}_${ff_schedulers[o]}.log
                                python ./core/vpns.py \
                                    --experiment_name ${experiment_name} \
                                    --dataset ${datasets[i]} \
                                    --network ${networks[j]} \
                                    --prune_method ${prune_methods[l]} \
                                    --prune_mode ${prune_modes[k]} \
                                    --label_mapping_mode ${label_mapping_modes[m]} \
                                    --ff_optimizer ${ff_optimizers[n]} \
                                    --ff_scheduler ${ff_schedulers[o]} \
                                    --gpu ${gpus[o]} \
                                    --epochs ${epochs} \
                                    --seed ${seed} \
                                    > $log_filename 2>&1 &
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
    wait
done
