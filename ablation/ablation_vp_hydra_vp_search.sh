#!/bin/sh

experiment_name='ablation_vp_hydra_vp_search'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# ['random', 'imp', 'omp', 'grasp', 'snip', 'synflow', 'hydra']
# dataset=('ucf101' 'cifar10' 'cifar100' 'svhn' 'mnist' 'flowers102')
networks=('resnet18')
datasets=('cifar10')
epochs=50
seed=7
prune_modes=('vp_ff')
prune_methods=('hydra')
density_list='1,0.01'
second_phases=('vp+ff_cotrain')

# input_sizes=(224 192 160 128 96 64 32)
#pad_sizes=(16 32 48 64 80 96 112)


input_sizes=(128)
pad_sizes=(16 32 48 64)
gpus=(7 6 5 4)
for j in ${!networks[@]};do
    for i in ${!datasets[@]};do
        for l in ${!prune_methods[@]};do
            for k in ${!input_sizes[@]};do
               for m in ${!pad_sizes[@]};do                    
                    log_filename=${foler_name}/${networks[j]}_${datasets[i]}_${prune_methods[l]}_${input_sizes[k]}_${pad_sizes[m]}_${seed}.log
                        python ./core/vpns_new.py \
                            --experiment_name ${experiment_name} \
                            --dataset ${datasets[i]} \
                            --network ${networks[j]} \
                            --prune_mode ${prune_modes[0]} \
                            --prune_method ${prune_methods[0]} \
                            --density_list ${density_list} \
                            --second_phase ${second_phases[0]} \
                            --input_size ${input_sizes[k]} \
                            --pad_size ${pad_sizes[m]} \
                            --gpu ${gpus[m]} \
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
