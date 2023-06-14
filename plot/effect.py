import torch
import os
import pickle


if __name__ == '__main__':
    dir_path = '/data4/hop20001/can/ILM-VP/results/vp_model_dataset/resnet18/cifar100/VPpad/FF0/LP0/PRUNEimp/LMilm/adam/LR0.01/multistep/EPOCHS200/IMAGESIZE128_48_183/SEED7/GPU7/DENSITY0.8'
    # with open(dir_path, 'rb') as file:
    #     data = pickle.load(file)
    # print(file)
    ckpt_path = []
    acc = []
    for i in range(0,10):
        ckpt_path.append(os.path.join(dir_path, str(i) + 'best.pth'))
    for path in ckpt_path:
        acc.append(torch.load(path)['ckpt_test_acc'])
    print(acc)

# 0614 
vp_ilm_cifar10 = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
vp_ilm_cifar100 = [0.4546, 0.4602, 0.4688, 0.4574, 0.4515, 0.4321, 0.4021, 0.3656, 0.3261, 0.2735]


# ablation vp_optimizer lr = 0.01
adam_cosine = 0.8304
adam_multistep = 0.8353
sgd_cosine = 0.7491
sgd_multistep = 0.7456


# ablation lm_interval
interval_1 = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
interval_5 = [0.7668, 0.8191, 0.8234, 0.7646, 0.7447, 0.7305, 0.6974, 0.6646, 0.6512, 0.6449]
interval_10 = [0.7779, 0.8205, 0.7903, 0.7408, 0.7363, 0.7028, 0.6532, 0.6344, 0.6429, 0.6205]
interval_15 = [0.7737, 0.8289, 0.7532, 0.7599, 0.7362, 0.6862, 0.683, 0.6598, 0.6308, 0.6278]
interval_20 = [0.7503, 0.7127, 0.7377, 0.7127, 0.667, 0.6458, 0.6462, 0.6547, 0.6393, 0.6136]


# 0612
# resnet50+cifar10
ff_flm_imp = [0.9766, 0.9731, 0.9758, 0.977]
notune_ilm_imp = [0.5721, 0.5711, 0.5277, 0.4724, 0.3405, 0.3105, 0.2687, 0.209, 0.1745, 0.1038]
notune_ilm_omp = [0.5721, 0.5693, 0.5284, 0.4807, 0.3591, 0.3386, 0.2708, 0.2162, 0.1815, 0.1018]

# resnet50+cifar100
notune_ilm_imp = [0.2975, 0.3129, 0.3007, 0.256, 0.1957, 0.1585, 0.0995, 0.0666, 0.0394, 0.0145]
notune_ilm_omp = [0.3036, 0.3005, 0.3011, 0.2574, 0.2087, 0.1626, 0.1073, 0.0689, 0.0402, 0.0121]

# resnet18+cifar100
ff_flm_imp = [0.8084, 0.8117, 0.8137, 0.8151, 0.8094, 0.8131, 0.81, 0.8021, 0.8014, 0.7971]
notune_ilm_imp = [0.2447, 0.2384, 0.2295, 0.2239, 0.2085, 0.1332, 0.0806, 0.0445, 0.0398, 0.0409]
notune_ilm_omp = [0.2411, 0.245, 0.2266, 0.2292, 0.1911, 0.1134, 0.0653, 0.0375, 0.0443, 0.0293]
vp_ilm_hydra = [0.1951, 0.7673, 0.7442, 0.7063, 0.6638, 0.6385, 0.6319, 0.6178]
vp_ilm_imp = [0.4546, 0.4602, 0.4688, 0.4574, 0.4515, 0.4321, 0.4021, 0.3656]
vp_ilm_omp = [0.1951, 0.4709, 0.4646, 0.4523, 0.4478, 0.4342, 0.3843, 0.3568]

# ablation vp/ff/no_tune/vp+ff
# do more experiments about ff_hydra(whether use scaled init), hydra_no_tune(lr 0.0001)
ff_flm_hydra = [0.96, 0.9124, 0.9119, 0.9031, 0.9018, 0.8994, 0.892, 0.885, 0.8765]
ff_flm_imp = [0.96, 0.9637, 0.96, 0.9627, 0.9639, 0.9629, 0.9608, 0.9594, 0.9589, 0.9603]
notune_ilm_hydra = [0.502, 0.9292]
notune_ilm_imp = [0.502, 0.4985, 0.4704, 0.4369, 0.3781, 0.2663, 0.2353, 0.1757, 0.1868, 0.1899]
notune_ilm_omp = [0.502, 0.4996, 0.4703, 0.4547, 0.3484, 0.2767, 0.1896, 0.1961, 0.2069, 0.1643]
vp_ff_flm_hydra = [0.9427, 0.8959, 0.895, 0.8872, 0.8827]
vp_ff_flm_imp = [0.9427, 0.9416, 0.9418, 0.9406]
vp_ilm_hydra = [0.8353, 0.9427, 0.9285, 0.9058, 0.8839, 0.8841, 0.8741, 0.8613, 0.8543, 0.8565]
vp_ilm_imp = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
vp_ilm_omp = [0.8353, 0.8088, 0.8112, 0.7888, 0.7536, 0.6904, 0.672, 0.6677, 0.6506, 0.6429]

# 0609
# ablation_vp_search_exp
inputsize32=[0.5042, 0.5693, 0.6122, 0.6350, 0.6471, 0.6765, 0.6672]
inputsize64=[0.5826, 0.6416, 0.6658, 0.6905, 0.7240, 0.6711, 0.6872]
inputsize96=[0.6171, 0.7187, 0.6886, 0.7693, 0.6825, 0.6783, 0.7053]
inputsize128=[0.6550, 0.7751, 0.8353, 0.6933, 0.6691, 0.6674, 0.7066]
inputsize160=[0.6631, 0.8051, 0.7328, 0.6551, 0.6581, 0.6611, 0.6803]
inputsize192=[0.7615, 0.7751, 0.6525, 0.6289, 0.6352, 0.6484, 0.6738]
inputsize224=[0.6291, 0.6060, 0.6425, 0.6177, 0.6140, 0.6209, 0.6258]


# ablation_prompt_methods_exp
pad = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
fix = [0.676, 0.666, 0.6682, 0.6465, 0.6557, 0.6449, 0.6284, 0.645, 0.6358, 0.6351]
random = [0.5752, 0.5892, 0.5831, 0.5999, 0.6031, 0.5896, 0.5866, 0.5687, 0.5706, 0.5824]

# ablation_no_sigmoid_exp
no_sigmoid = [0.7628, 0.8205, 0.7996, 0.7786, 0.6982, 0.6898, 0.6721, 0.6788, 0.6593, 0.6578]
sigmoid = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]

# ablation_lm_methods_exp
flm = [0.7425, 0.7348, 0.7421, 0.7325, 0.728, 0.7253, 0.718, 0.6968, 0.694, 0.666]
ilm = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]

# ablation_hydra_init_exp
init_1 = [0.7948, 0.9407, 0.9319, 0.9148, 0.8877, 0.8754, 0.8754, 0.8641, 0.8549, 0.8427]
init_0 = [0.7948, 0.9398, 0.9238, 0.9092, 0.8918, 0.8681, 0.8674, 0.8614, 0.8569, 0.8484]

# ablation_randomcrop_exp
crop_0=[0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
crop_1=[0.8358, 0.8212, 0.8183, 0.7219, 0.7391, 0.6887, 0.6877, 0.6386, 0.6599, 0.6235]

# 0608
# ablation_hydra_init_exp
init_1 = [0.9497, 0.9395, 0.9263, 0.9059, 0.8928]
init_0 = [0.9489, 0.9364, 0.9222, 0.9002, 0.8836]

# ablation_randomcrop_exp
crop_0=[0.8152, 0.8186, 0.8265, 0.801, 0.7658]
crop_1=[0.8271, 0.8174, 0.818, 0.7829, 0.7645]

# 0605
# prompt_prune_exp
imp_flm = [0.7401, 0.744, 0.738, 0.7435, 0.7239, 0.7185, 0.7072, 0.6968, 0.682, 0.6548]
imp_ilm = [0.8051, 0.8267, 0.813, 0.8159, 0.7851, 0.6392, 0.6771, 0.6365, 0.6127, 0.6011]
omp_flm = [0.7401, 0.7415, 0.7391, 0.734, 0.7254, 0.71, 0.7018, 0.6998, 0.662, 0.6314]
omp_ilm = [0.8051, 0.8308, 0.7939, 0.7929, 0.7799, 0.6419, 0.6186, 0.6202, 0.6103, 0.5916]
grasp_flm = [0.7401, 0.6009, 0.5669, 0.5299, 0.2504, 0.1938, 0.2389, 0.2993, 0.1836, 0.2076]
grasp_ilm = [0.8051, 0.5622, 0.5446, 0.5231, 0.4542, 0.4376, 0.3994, 0.3692, 0.3823, 0.3865]
hydra_flm = [0.7401, 0.9491, 0.9395, 0.9157, 0.905, 0.884, 0.8761, 0.8713, 0.866, 0.864]
hydra_ilm = [0.8051, 0.9489, 0.9364, 0.9222, 0.9002, 0.8836, 0.8768, 0.8759, 0.8732, 0.8649]

# tricks_exp_flm_after_prune_notune
imp = [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]
omp = [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]
grasp = [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]


# tricks_exp_flm_after_prune
omp_flm_pre = [0.4614, 0.7415, 0.7391]
omp_flm_after = [0.4616, 0.7378, 0.75]
omp_ilm = [0.4614, 0.8308, 0.7939]
grasp_flm_pre = [0.4616, 0.6009, 0.5669]
grasp_flm_after = [0.4616, 0.5506, 0.522]
grasp_ilm = [0.4616, 0.5622, 0.5446]

# prompt_loc_exp
pad = [0.6286, 0.6224]
fix = [0.5754, 0.5814]
random = [0.4755, 0.4678]






# 0604
input_32 = [0.5522, 0.5565, 0.5599, 0.5636, 0.5581]
input_64 = [0.6043, 0.6007, 0.6074, 0.6046, 0.5961]
input_96 = [0.6374, 0.6385, 0.6403, 0.6382, 0.635]
input_128 = [0.6492, 0.6542, 0.661, 0.6453, 0.6454]
input_160 = [0.7401, 0.744, 0.738, 0.7435, 0.7239]
input_192 = [0.6773, 0.6789, 0.68, 0.6758, 0.6689]
input_224 = [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]



pad_112 = [0.6179, 0.6147, 0.6077, 0.6173, 0.6145]
pad_96 = [0.6118, 0.6199, 0.6159, 0.6226, 0.6082]
pad_80 = [0.6166, 0.6202, 0.6142, 0.6115, 0.6159]
pad_64 = [0.6135, 0.6112, 0.6037, 0.6072, 0.61]
pad_48 = [0.614, 0.6055, 0.6066, 0.6072, 0.5923]
pad_32 = [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]
pad_16 = [0.5896, 0.5918, 0.5919, 0.5892, 0.5845]
# pad_16 0.5894
# pad_32 0.6157600000000001
# pad_64 0.60912
# pad_80 0.61568
# pad_96 0.61568
# pad_112 0.6144200000000001

# 0603
# prune without tune
imp_flm = [0.5042, 0.4983, 0.4723, 0.4754, 0.4616, 0.455, 0.3472, 0.1568, 0.1268, 0.0971]
imp_ilm = [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]
omp_flm = [0.5042, 0.499, 0.475, 0.4825, 0.4797, 0.4214, 0.2975, 0.1296, 0.1028, 0.1003]
omp_ilm = [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]
grasp_flm = [0.5042, 0.0848, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
grasp_ilm = [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]
hydra_flm = [0.5042, 0.1104, 0.1149, 0.0976, 0.1053, 0.1, 0.1, 0.1, 0.1, 0.1]
hydra_ilm = [0.5042, 0.1724, 0.1217, 0.1355, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]




# 0602
# TODO initialization may be different
imp_ff_flm = [0.9625, 0.9668, 0.9652, 0.9656, 0.9676, 0.9668, 0.9651, 0.9672, 0.9649, 0.9648]
imp_ff_ilm = [0.9625, 0.9666, 0.9682, 0.9655, 0.966, 0.9661, 0.9645, 0.9638, 0.9652, 0.962]
omp_ff_flm = [0.9625, 0.9633, 0.9627, 0.9615, 0.9604, 0.9586, 0.9581, 0.9582, 0.9563, 0.9552]
omp_ff_ilm = [0.9625, 0.9632, 0.9635, 0.9621, 0.9631, 0.9586, 0.9592, 0.9582, 0.9541, 0.9529]
grasp_ff_flm = [0.9625, 0.9602, 0.9585, 0.9532, 0.9526, 0.9451, 0.9406, 0.9383, 0.9386, 0.9322]
grasp_ff_ilm = [0.9625, 0.9599, 0.9544, 0.9511, 0.9483, 0.9453, 0.944, 0.9399, 0.9375, 0.9321]
hydra_ff_flm = [0.9625, 0.9628, 0.9592, 0.9544, 0.9475, 0.9383, 0.9276, 0.9246, 0.9183, 0.9201]
hydra_ff_ilm = [0.9625, 0.9636, 0.9579, 0.9564, 0.9449, 0.9363, 0.9288, 0.9258, 0.923, 0.9193]

imp_vp_flm = [0.6528, 0.6181, 0.617, 0.6218, 0.6146, 0.6123, 0.6003, 0.5842, 0.5638, 0.5361]
imp_vp_ilm = [0.66, 0.6164, 0.6144, 0.6151, 0.6131, 0.6008, 0.5878, 0.5705, 0.5475, 0.5183]
omp_vp_flm = [0.6528, 0.616, 0.621, 0.6132, 0.6179, 0.6121, 0.6012, 0.571, 0.5351, 0.4976]
omp_vp_ilm = [0.66, 0.6265, 0.6261, 0.6157, 0.6067, 0.5925, 0.5672, 0.549, 0.5024, 0.5011]
grasp_vp_flm = [0.6325, 0.5462, 0.5359, 0.5493, 0.5429, 0.5565, 0.5466, 0.2311, 0.2327]
grasp_vp_ilm = [0.5684, 0.5709, 0.5428, 0.5398, 0.5448, 0.5711, 0.5737, 0.4063, 0.4035]
hydra_vp_flm = [0.8399, 0.8215, 0.8102, 0.8056, 0.8002, 0.7837, 0.7844, 0.7703, 0.768]
hydra_vp_ilm = [0.8314, 0.8239, 0.809, 0.798, 0.7884, 0.7923, 0.786, 0.7725, 0.752]


pad_16 = [0.551, 0.5445, 0.5473, 0.5505, 0.5392, 0.5379, 0.521, 0.5333, 0.5246, 0.5084]
pad_32 = [0.6159, 0.6133, 0.6148, 0.613, 0.6143, 0.6064, 0.5956, 0.5845, 0.5803, 0.5809]
pad_48 = [0.6256, 0.6252, 0.6327, 0.6248, 0.6276, 0.6132, 0.6179, 0.6032, 0.6079, 0.601]
pad_64 = [0.6273, 0.6332, 0.6246, 0.6243, 0.6259, 0.6268, 0.6215, 0.6224, 0.6175, 0.6117]
pad_80 = [0.6356, 0.6414, 0.6382, 0.6296, 0.6417, 0.628, 0.6266, 0.6269, 0.6227, 0.6298]
pad_96 = [0.6442, 0.6357, 0.6494, 0.6326, 0.6388, 0.636, 0.6266, 0.6288, 0.6333, 0.624]


