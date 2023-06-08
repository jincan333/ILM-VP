#!/bin/sh

# parser.add_argument('--experiment_name', default='exp', type=str, help='name of experiment, the save directory will be save_dir+exp_name')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--label_mapping_mode', type=str, default='ilm', help='label mapping methods: rlm, flm, ilm, None', choices=['flm', 'ilm', 'rlm', None])
# parser.add_argument('--prompt_method', type=str, default='pad', help='None, expand, pad, fix, random', choices=['expand', 'pad', 'fix', 'random', None])
# parser.add_argument('--input_size', type=int, default=160, help='image size before prompt, no more than 224', choices=[224, 192, 160, 128, 96, 64, 32])
# parser.add_argument('--pad_size', type=int, default=32, help='only for padprompt, no more than 112, parameters cnt 4*pad**2+896pad', choices=[0, 16, 32, 48, 64, 80, 96, 112])
# parser.add_argument('--mask_size', type=int, default=96, help='only for fixadd and randomadd, no more than 224, parameters cnt mask**2', choices=[115, 156, 183, 202, 214, 221, 224])
# parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: sgd. Options: sgd, adam.', choices=['sgd', 'adam'])
# parser.add_argument('--lr_scheduler', default='multistep', help='decreasing strategy. Default: cosine, multistep', choices=['cosine', 'multistep'])
# parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
# parser.add_argument('--prune_method', type=str, default='hydra', help='prune methods: imp, omp, grasp, hydra', choices=['imp', 'omp', 'grasp', 'hydra'])
# parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
# parser.add_argument('--density', type=float, default=0.80, help='The density of the overall sparse network.')
# parser.add_argument('--hydra_scaled_init', type=int, default=0, help='whether use scaled initialization for hydra or not.', choices=[0, 1])
# parser.add_argument('--train_model', type=int, default=1, help='whether training the model.', choices=[0, 1])
# parser.add_argument('--flm_loc', type=str, default='pre', help='pre-train flm or after-prune flm.', choices=['pre', 'after'])
# parser.add_argument('--randomcrop', type=int, default=0, help='dataset randomcrop.', choices=[0, 1])
# parser.add_argument('--seed', default=7, type=int, help='random seed')

experiment_name='prune_vp_exp'
if [ ! -d ${experiment_name} ]; then
    mkdir -p ${experiment_name}
fi

prune_methods=('imp' 'omp' 'hydra')
label_mapping_modes=('ilm')
prompt_methods=('pad')
gpus=(0 1 2)
input_sizes=(160)
pad_sizes=(32)
pruning_times=3
epochs=3
seed=7
for i in ${!prune_methods[@]};do
    log_filename=${experiment_name}/prune_method_${prune_methods[i]}.log
    python ./experiments/prompt_sparsity/lm_vp_prune.py \
        --experiment_name ${experiment_name} \
        --prune_method ${prune_methods[i]} \
        --label_mapping_mode ${label_mapping_modes[0]} \
        --prompt_method ${prompt_methods[0]} \
        --gpu ${gpus[i]} \
        --input_size ${input_sizes[0]} \
        --pad_size ${pad_sizes[0]} \
        --pruning_times ${pruning_times} \
        --epochs ${epochs} \
        --seed ${seed} \
        > $log_filename 2>&1 &
done