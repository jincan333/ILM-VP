#!/bin/sh

# pad size experiment pad_sizes=(16 32 48 64 80 96 112) input_size=224 gpus=(0 0 0 0 0 0 0)
# exp_name='padsize_exp'
# pad_sizes=(16 32 48 64 80 96 112)
# gpus=(0 0 0 0 0 0 0)
# for i in ${!pad_sizes[@]}; 
# do
#     log_filename='${exp_name}_224_${pad_sizes[i]}.log'
#     python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name $exp_name --gpu ${gpus[i]} --input_size 224 --pad_size ${pad_sizes[i]} --epochs 200 > $log_filename 2>&1
# done

# input size experiment input_sizes=(224 192 160 128 96 64 32) pad_size=(16 32 48 64 80 96 112) gpus=(7 6 5 4 3 2 1)
input_sizes=(192 160 128 96 64 32)
gpus=(7 6 5 4 3 2)
pad_sizes=(64 80 96 112)
for pad_size in ${pad_sizes[@]};do
    for i in ${!input_sizes[@]};do
        log_filename=inputsize_${input_sizes[i]}_${pad_size}.log
        python ./experiments/prompt_sparsity/lm_vp_prune.py --experiment_name inputsize_exp --gpu ${gpus[i]} --input_size ${input_sizes[i]} --pad_size ${pad_size} --epochs 200 > $log_filename 2>&1 &
    done
    wait
done





# parser.add_argument('--experiment_name', default='inputsize_exp', type=str, help='name of experiment, the save directory will be save_dir+exp_name')
# parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
# parser.add_argument('--label_mapping_mode', type=str, default='flm', help='label mapping methods: rlm, flm, ilm, None')
# parser.add_argument('--prompt_method', type=str, default='pad', help='None, expand, pad, fix, random')
# parser.add_argument('--input_size', type=int, default=192, help='image size before prompt, no more than 224', choices=[224, 192, 160, 128, 96, 64, 32])
# parser.add_argument('--pad_size', type=int, default=112, help='only for padprompt, no more than 112, parameters cnt 4*pad**2+896pad', choices=[16, 32, 48, 64, 80, 96, 112])
# parser.add_argument('--mask_size', type=int, default=96, help='only for fixadd and randomadd, no more than 224, parameters cnt mask**2', choices=[115, 156, 183, 202, 214, 221, 224])
# parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
# parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy. Default: cosine, multistep')
# parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
# parser.add_argument('--prune_method', type=str, default='imp', help='prune methods: imp, omp, grasp, hydra')
# parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
# parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')