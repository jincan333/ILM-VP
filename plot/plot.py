import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import os
import datetime
import numpy as np

# 0731
# combine visual prompting with current methods
snip = [72.80, 36.38, 8.29, 1.27]
grasp = [72.80, 45.16, 33.28, 9.44]
synflow = [72.80, 56.66, 34.82, 5.16]
omp = [72.80, 62.81, 33.66, 1.31]
vpns = [72.80, 66.75, 54.02, 11.50]

save_dir = os.path.join('results/combine visual prompting with current methods', str(datetime.datetime.now().date()))
image_name = 'Combine Visual Prompting'
os.makedirs(save_dir, exist_ok=True)
results = {
    'Compression Ratio': [1, 10, 100, 1000]
    ,'VP+SNIP': snip
    ,'VP+GraSP': grasp
    ,'VP+SynFlow': synflow
    ,'VP+OMP': omp
    ,'VP+VPNs': vpns
}
plt.plot(results['Compression Ratio'], results['VP+SNIP'], label='VP+SNIP', color=(0.902, 0.621, 0), linestyle='-', marker='o', linewidth=2, markersize=8)
plt.plot(results['Compression Ratio'], results['VP+GraSP'], label='VP+GraSP', color=(0.357, 0.714, 0.914), linestyle='-', marker='o', linewidth=2, markersize=8)
plt.plot(results['Compression Ratio'], results['VP+SynFlow'], label='VP+SynFlow', linestyle='-', marker='o', linewidth=2, markersize=8)
plt.plot(results['Compression Ratio'], results['VP+OMP'], label='VP+OMP', linestyle='-', marker='o', linewidth=2, markersize=8)
plt.plot(results['Compression Ratio'], results['VP+VPNs'], label='VP+VPNs', linestyle='-', marker='o', linewidth=2, markersize=8)
# plt.title(image_name)
plt.gca().set_xscale('log')
plt.xlabel('Compression Ratio')
plt.ylabel('Accuracy(%)')
plt.grid()
plt.legend(loc='lower left', prop={'size':8})
plt.tight_layout()
plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
plt.close()


# 0730
# transferability over datasets
# CIFAR100
# save_dir = os.path.join('results/transferability over datasets', str(datetime.datetime.now().date()))
# os.makedirs(save_dir, exist_ok=True)
# image_name = 'Dataset Transfer CIFAR100'
# models = ['VPNs', 'HYDRA', 'OMP']
# compression_ratios = ['90%', '99%', '99.9%']
# values = np.array([[100, 100, 100], [77.42, 65.30, 17.69], [75.06, 46.59, 6.15]])
# x = np.arange(len(compression_ratios))
# width = 0.2
# fig, ax = plt.subplots(figsize=(10, 5))
# for i in range(len(models)):
#     ax.bar(x + i*width - width, values[i], width, label=models[i])
# ax.axhline(y=81.64, color='black', linestyle='dotted')
# ax.set_xlabel('Sparsity')
# ax.set_ylabel('Accuracy(%)')
# ax.set_xticks(x)
# ax.set_xticklabels(compression_ratios)
# leg1 = ax.legend(loc='upper right')
# ax.add_artist(leg1)
# dense_line = mlines.Line2D([], [], color='black', linestyle='dotted', markersize=7, label='Dense')
# leg2 = ax.legend(handles=[dense_line], bbox_to_anchor=(1, 0.83), loc='upper right')
# ax.add_artist(leg2)
# plt.title('CIFAR100')
# fig.tight_layout()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()


# Tiny ImageNet
# save_dir = os.path.join('results/transferability over datasets', str(datetime.datetime.now().date()))
# os.makedirs(save_dir, exist_ok=True)
# image_name = 'Dataset Transfer Tiny ImageNet'
# models = ['VPNs', 'HYDRA', 'OMP']
# compression_ratios = ['90%', '99%', '99.9%']
# values = np.array([[100, 100, 100], [66.43, 46.73, 7.07], [62.41, 33.04, 2.19]])
# x = np.arange(len(compression_ratios))
# width = 0.2
# fig, ax = plt.subplots(figsize=(10, 5))
# for i in range(len(models)):
#     ax.bar(x + i*width - width, values[i], width, label=models[i])
# ax.axhline(y=72.08, color='black', linestyle='dotted')
# ax.set_xlabel('Sparsity')
# ax.set_ylabel('Accuracy(%)')
# ax.set_xticks(x)
# ax.set_xticklabels(compression_ratios)
# leg1 = ax.legend(loc='upper right')
# ax.add_artist(leg1)
# dense_line = mlines.Line2D([], [], color='black', linestyle='dotted', markersize=7, label='Dense')
# leg2 = ax.legend(handles=[dense_line], bbox_to_anchor=(1, 0.83), loc='upper right')
# ax.add_artist(leg2)
# plt.title('Tiny ImageNet')
# fig.tight_layout()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()


# transferability over architectures
# resnet18_vpns = [72.80, 66.75, 54.02, 11.50]
# resnet18_hydra = [72.80, 66.43, 46.73, 7.07]
# resnet50_vpns = [83.40, 70.52, 59.80, 0]
# resnet50_hydra = [83.40, 68.92, 55.86, 11.23]
# vgg_vpns = [70.19, 50.50, 24.92, 0]
# vgg_hydra = [70.19, 50.07, 0, 0]

# save_dir = os.path.join('results/transferability over architectures', str(datetime.datetime.now().date()))
# image_name = 'Architectures Transfer'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'Compression Ratio': [1, 10, 100, 1000]
#     ,'ResNet18-VPNs': resnet18_vpns
#     ,'ResNet18-HYDRA': resnet18_hydra
#     ,'ResNet50-VPNs': resnet50_vpns
#     ,'ResNet50-HYDRA': resnet50_hydra
#     ,'VGG16-VPNs': vgg_vpns
#     ,'VGG16-HYDRA': vgg_hydra
# }
# line_res18 = mlines.Line2D([], [], color=(0.902, 0.621, 0), linestyle='-', markersize=7, label="ResNet-18")
# line_res50 = mlines.Line2D([], [], color=(0.357, 0.714, 0.914), linestyle='-', markersize=7, label="ResNet-50")
# line_vgg    = mlines.Line2D([], [], color='red', linestyle='-', markersize=7, label="VGG-16")
# line_vpns   = mlines.Line2D([], [], color='black', marker='x', linestyle='-', markersize=7, label='VPNs')
# line_hydra  = mlines.Line2D([], [], color='black', marker='*', linestyle=':', markersize=7, label='HYDRA')
# plt.figure(figsize=(10, 5))
# plt.plot(results['Compression Ratio'], results['ResNet18-VPNs'], color=(0.902, 0.621, 0), linestyle='-', marker='x', linewidth=2, markersize=8)
# plt.plot(results['Compression Ratio'], results['ResNet18-HYDRA'], color=(0.902, 0.621, 0), linestyle=':', marker='*', linewidth=2, markersize=8)
# plt.plot(results['Compression Ratio'], results['ResNet50-VPNs'], color=(0.357, 0.714, 0.914), linestyle='-', marker='x', linewidth=2, markersize=8)
# plt.plot(results['Compression Ratio'], results['ResNet50-HYDRA'], color=(0.357, 0.714, 0.914), linestyle=':', marker='*', linewidth=2, markersize=8)
# plt.plot(results['Compression Ratio'], results['VGG16-VPNs'], color='r', linestyle='-', marker='x', linewidth=2, markersize=8)
# plt.plot(results['Compression Ratio'], results['VGG16-HYDRA'], color='r', linestyle=':', marker='*', linewidth=2, markersize=8)
# # plt.title(image_name)
# plt.gca().set_xscale('log')
# plt.xlabel('Compression Ratio')
# plt.ylabel('Accuracy(%)')
# legend_arch = plt.legend(handles=[line_res18, line_res50, line_vgg], loc='upper right', bbox_to_anchor=(1, 1), prop={'size':8})
# legend_style =  plt.legend(handles=[line_vpns, line_hydra], loc='upper right', bbox_to_anchor=(1, 0.85), prop={'size':8})
# plt.gca().add_artist(legend_arch)
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))


# 0614
# ablation lm_interval
# interval_1 = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
# interval_5 = [0.7668, 0.8191, 0.8234, 0.7646, 0.7447, 0.7305, 0.6974, 0.6646, 0.6512, 0.6449]
# interval_10 = [0.7779, 0.8205, 0.7903, 0.7408, 0.7363, 0.7028, 0.6532, 0.6344, 0.6429, 0.6205]
# interval_15 = [0.7737, 0.8289, 0.7532, 0.7599, 0.7362, 0.6862, 0.683, 0.6598, 0.6308, 0.6278]
# interval_20 = [0.7503, 0.7127, 0.7377, 0.7127, 0.667, 0.6458, 0.6462, 0.6547, 0.6393, 0.6136]
# flm = [0.7425, 0.7348, 0.7421, 0.7325, 0.728, 0.7253, 0.718, 0.6968, 0.694, 0.666]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Label Mapping Interval'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'interval_1': [_*100 for _ in interval_1]
#     ,'interval_5': [_*100 for _ in interval_5]
#     ,'interval_10': [_*100 for _ in interval_10]
#     ,'interval_15': [_*100 for _ in interval_15]
#     ,'interval_20': [_*100 for _ in interval_20]
#     ,'flm': [_*100 for _ in flm]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'][:len(results[k])], results[k], marker='o', markersize=4, label=k)

# plt.title(image_name)
# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=4, prop={'size':8})
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))



# 0612
# resnet18+cifar100
# notune_imp = [0.2447, 0.2384, 0.2295, 0.2239, 0.2085, 0.1332, 0.0806, 0.0445, 0.0398, 0.0409]
# notune_omp = [0.2411, 0.245, 0.2266, 0.2292, 0.1911, 0.1134, 0.0653, 0.0375, 0.0443, 0.0293]
# vp_hydra = [0.4546, 0.7673, 0.7442, 0.7063, 0.6638, 0.6385, 0.6319, 0.6178]
# vp_imp = [0.4546, 0.4602, 0.4688, 0.4574, 0.4515, 0.4321, 0.4021, 0.3656]
# vp_omp = [0.4546, 0.4709, 0.4646, 0.4523, 0.4478, 0.4342, 0.3843, 0.3568]


# save_dir = os.path.join('results/superior_performance', str(datetime.datetime.now().date()))
# image_name = 'ResNet18+CIFAR100'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'notune_imp': [_*100 for _ in notune_imp]
#     ,'notune_omp': [_*100 for _ in notune_omp]
#     ,'vp_hydra': [_*100 for _ in vp_hydra]
#     ,'vp_imp': [_*100 for _ in vp_imp]
#     ,'vp_omp': [_*100 for _ in vp_omp]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'][:len(results[k])], results[k], marker='o', markersize=4, label=k)

# plt.title(image_name)
# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=4, prop={'size':8})
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))


# ablation ff vs vp+ff
# ff_hydra = [0.96, 0.9628, 0.9592, 0.9544, 0.9475, 0.9383, 0.9276, 0.9246, 0.9183, 0.9201]
# ff_imp = [0.96, 0.9637, 0.96, 0.9627, 0.9639, 0.9629, 0.9608, 0.9594, 0.9589, 0.9603]
# vp_ff_hydra = [0.9427, 0.8959, 0.895, 0.8872, 0.8827]
# vp_ff_imp = [0.9427, 0.9416, 0.9418, 0.9406]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'FF vs VP+FF - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'ff_hydra': [_*100 for _ in ff_hydra]
#     ,'ff_imp': [_*100 for _ in ff_imp]
#     ,'vp_ff_hydra': [_*100 for _ in vp_ff_hydra]
#     ,'vp_ff_imp': [_*100 for _ in vp_ff_imp]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'][:len(results[k])], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=4)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# ablation vp vs notune

# notune_hydra = [0.502, 0.9292]
# notune_imp = [0.502, 0.4985, 0.4704, 0.4369, 0.3781, 0.2663, 0.2353, 0.1757, 0.1868, 0.1899]
# notune_omp = [0.502, 0.4996, 0.4703, 0.4547, 0.3484, 0.2767, 0.1896, 0.1961, 0.2069, 0.1643]
# vp_hydra = [0.8353, 0.9427, 0.9285, 0.9058, 0.8839, 0.8841, 0.8741, 0.8613, 0.8543, 0.8565]
# vp_imp = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
# vp_omp = [0.8353, 0.8088, 0.8112, 0.7888, 0.7536, 0.6904, 0.672, 0.6677, 0.6506, 0.6429]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'VP vs NoTune - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'notune_hydra': [_*100 for _ in notune_hydra]
#     ,'notune_imp': [_*100 for _ in notune_imp]
#     ,'notune_omp': [_*100 for _ in notune_omp]
#     ,'vp_hydra': [_*100 for _ in vp_hydra]
#     ,'vp_imp': [_*100 for _ in vp_imp]
#     ,'vp_omp': [_*100 for _ in vp_omp]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'][:len(results[k])], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=4)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))



# 0610
# ablation vp/ff/no_tune/vp+ff
# scaled_initilization = [0.8353, 0.9407, 0.9319, 0.9148, 0.8877, 0.8754, 0.8754, 0.8641, 0.8549, 0.8427]
# kaiming_uniform = [0.8353, 0.9398, 0.9238, 0.9092, 0.8918, 0.8681, 0.8674, 0.8614, 0.8569, 0.8484]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Hydra Score Initialization - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'scaled_initilization': [_*100 for _ in scaled_initilization]
#     ,'kaiming_uniform': [_*100 for _ in kaiming_uniform]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))


# ablation hydra init
# scaled_initilization = [0.8353, 0.9407, 0.9319, 0.9148, 0.8877, 0.8754, 0.8754, 0.8641, 0.8549, 0.8427]
# kaiming_uniform = [0.8353, 0.9398, 0.9238, 0.9092, 0.8918, 0.8681, 0.8674, 0.8614, 0.8569, 0.8484]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Hydra Score Initialization - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'scaled_initilization': [_*100 for _ in scaled_initilization]
#     ,'kaiming_uniform': [_*100 for _ in kaiming_uniform]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# ablation randomcrop
# no_randomcrop=[0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
# randomcrop=[0.8358, 0.8212, 0.8183, 0.7219, 0.7391, 0.6887, 0.6877, 0.6386, 0.6599, 0.6235]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Randomcrop - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'randomcrop': [_*100 for _ in randomcrop]
#     ,'no_randomcrop': [_*100 for _ in no_randomcrop]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# ablation sigmoid
# no_sigmoid = [0.7628, 0.8205, 0.7996, 0.7786, 0.6982, 0.6898, 0.6721, 0.6788, 0.6593, 0.6578]
# sigmoid = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Sigmoid - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'sigmoid': [_*100 for _ in sigmoid]
#     ,'no_sigmoid': [_*100 for _ in no_sigmoid]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# ablation label mapping methods
# flm = [0.7425, 0.7348, 0.7421, 0.7325, 0.728, 0.7253, 0.718, 0.6968, 0.694, 0.666]
# ilm = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
# rlm = []

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Label Mapping Method - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'flm': [_*100 for _ in flm]
#     ,'ilm': [_*100 for _ in ilm]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# ablation prompt methods
# pad = [0.8353, 0.8111, 0.826, 0.7954, 0.7397, 0.6795, 0.6741, 0.6481, 0.6619, 0.6377]
# fix_mask = [0.676, 0.666, 0.6682, 0.6465, 0.6557, 0.6449, 0.6284, 0.645, 0.6358, 0.6351]
# random_mask = [0.5752, 0.5892, 0.5831, 0.5999, 0.6031, 0.5896, 0.5866, 0.5687, 0.5706, 0.5824]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Prompt Method - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'pad': [_*100 for _ in pad]
#     ,'fix': [_*100 for _ in fix_mask]
#     ,'random': [_*100 for _ in random_mask]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# ablation promptsize
# inputsize32=[0.5042, 0.5693, 0.6122, 0.6350, 0.6471, 0.6765, 0.6672]
# inputsize64=[0.5826, 0.6416, 0.6658, 0.6905, 0.7240, 0.6711, 0.6872]
# inputsize96=[0.6171, 0.7187, 0.6886, 0.7693, 0.6825, 0.6783, 0.7053]
# inputsize128=[0.6550, 0.7751, 0.8353, 0.6933, 0.6691, 0.6674, 0.7066]
# inputsize160=[0.6631, 0.8051, 0.7328, 0.6551, 0.6581, 0.6611, 0.6803]
# inputsize192=[0.7615, 0.7751, 0.6525, 0.6289, 0.6352, 0.6484, 0.6738]
# inputsize224=[0.6291, 0.6060, 0.6425, 0.6177, 0.6140, 0.6209, 0.6258]

# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Padsize - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'padsize': [16, 32, 48, 64, 80, 96, 112]
#     ,'input32': [_*100 for _ in inputsize32]
#     ,'input64': [_*100 for _ in inputsize64]
#     ,'input96': [_*100 for _ in inputsize96]
#     ,'input128': [_*100 for _ in inputsize128]
#     ,'input160': [_*100 for _ in inputsize160]
#     ,'input192': [_*100 for _ in inputsize192]
#     ,'input224': [_*100 for _ in inputsize224]
# }
# for k in results.keys():
#     if k != 'padsize':
#         plt.plot(results['padsize'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Pad size')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=4, prop={'size':8})
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# ablation inputsize

# padsize16=[0.5042, 0.5826, 0.6171, 0.6550, 0.6631, 0.7615, 0.6291]
# padsize32=[0.5693, 0.6416, 0.7187, 0.7751, 0.8051, 0.7751, 0.6060]
# padsize48=[0.6122, 0.6658, 0.6886, 0.8353, 0.7328, 0.6525, 0.6425]
# padsize64=[0.6350, 0.6905, 0.7693, 0.6933, 0.6551, 0.6289, 0.6177]
# padsize80=[0.6471, 0.7240, 0.6825, 0.6691, 0.6581, 0.6352, 0.6140]
# padsize96=[0.6765, 0.6711, 0.6783, 0.6674, 0.6611, 0.6484, 0.6209]
# padsize112=[0.6672, 0.6872, 0.7053, 0.7066, 0.6803, 0.6738, 0.6258]
# save_dir = os.path.join('results/ablation_plot', str(datetime.datetime.now().date()))
# image_name = 'Inputsize - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'inputsize': [32, 64, 96, 128, 160, 192, 224]
#     ,'pad16': [_*100 for _ in padsize16]
#     ,'pad32': [_*100 for _ in padsize32]
#     ,'pad48': [_*100 for _ in padsize48]
#     ,'pad64': [_*100 for _ in padsize64]
#     ,'pad80': [_*100 for _ in padsize80]
#     ,'pad96': [_*100 for _ in padsize96]
#     ,'pad112': [_*100 for _ in padsize112]
# }
# for k in results.keys():
#     if k != 'inputsize':
#         plt.plot(results['inputsize'], results[k], marker='o', markersize=4, label=k)

# plt.xlabel('Input size')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=4, prop={'size':8})
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.pdf'), format='pdf')
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))


# 0605
# vp grasp+hydra
# grasp_flm = [0.7401, 0.6009, 0.5669, 0.5299, 0.2504, 0.1938, 0.2389, 0.2993, 0.1836, 0.2076]
# grasp_ilm = [0.8051, 0.5622, 0.5446, 0.5231, 0.4542, 0.4376, 0.3994, 0.3692, 0.3823]
# hydra_flm = [0.7401, 0.9491, 0.9395, 0.9157, 0.905, 0.884, 0.8761, 0.8713, 0.866, 0.864]
# hydra_ilm = [0.8051, 0.9489, 0.9364, 0.9222, 0.9002, 0.8836, 0.8768, 0.8759, 0.8732]

# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'VP GraSP+Hydra - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'GraSP-FLM': [_*100 for _ in grasp_flm]
#     ,'GraSP-ILM': [_*100 for _ in grasp_ilm]
#     ,'Hydra-FLM': [_*100 for _ in hydra_flm]
#     ,'Hydra-ILM': [_*100 for _ in hydra_ilm]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'][:len(results[k])], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()

# prompt_prune_exp
# imp_flm = [0.7401, 0.744, 0.738, 0.7435, 0.7239, 0.7185, 0.7072, 0.6968, 0.682, 0.6548]
# imp_ilm = [0.8051, 0.8267, 0.813, 0.8159, 0.7851, 0.6392, 0.6771, 0.6365, 0, 0]
# omp_flm = [0.7401, 0.7415, 0.7391, 0.734, 0.7254, 0.71, 0.7018, 0.6998, 0.662, 0.6314]
# omp_ilm = [0.8051, 0.8308, 0.7939, 0.7929, 0.7799, 0.6419, 0.6186, 0.6202, 0, 0]
# grasp_flm = [0.7401, 0.6009, 0.5669, 0.5299, 0.2504, 0.1938, 0.2389, 0.2993, 0.1836, 0.2076]
# grasp_ilm = [0.8051, 0.5622, 0.5446, 0.5231, 0.4542, 0.4376, 0.3994, 0.3692, 0.3823, 0]
# hydra_flm = [0.7401, 0.9491, 0.9395, 0.9157, 0.905, 0.884, 0.8761, 0.8713, 0.866, 0.864]
# hydra_ilm = [0.8051, 0.9489, 0.9364, 0.9222, 0.9002, 0.8836, 0.8768, 0.8759, 0.8732, 0]

# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'VP Prune - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'IMP-FLM': [_*100 for _ in imp_flm]
#     ,'IMP-ILM': [_*100 for _ in imp_ilm]
#     ,'OMP-FLM': [_*100 for _ in omp_flm]
#     ,'OMP-ILM': [_*100 for _ in omp_ilm]
#     ,'GraSP-FLM': [_*100 for _ in grasp_flm]
#     ,'GraSP-ILM': [_*100 for _ in grasp_ilm]
#     ,'Hydra-FLM': [_*100 for _ in hydra_flm]
#     ,'Hydra-ILM': [_*100 for _ in hydra_ilm]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'][:len(results[k])], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))



# tricks_exp_flm_after_prune_notune
# imp_flm_pre = [0.5042, 0.4983, 0.4723, 0.4754, 0.4616, 0.455, 0.3472, 0.1568, 0.1268, 0.0971]
# imp_flm_after = [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]
# imp_ilm = [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]
# omp_flm_pre = [0.5042, 0.499, 0.475, 0.4825, 0.4797, 0.4214, 0.2975, 0.1296, 0.1028, 0.1003]
# omp_flm_after = [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]
# omp_ilm = [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]
# grasp_flm_pre = [0.5042, 0.0848, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# grasp_flm_after = [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]
# grasp_ilm = [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]


# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'Without tune diff LM Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'imp_flm_pre': [_*100 for _ in imp_flm_pre]
#     ,'imp_flm_after': [_*100 for _ in imp_flm_after]
#     ,'imp_ilm': [_*100 for _ in imp_ilm]
#     ,'omp_flm_pre': [_*100 for _ in omp_flm_pre]
#     ,'omp_flm_after': [_*100 for _ in omp_flm_after]
#     ,'omp_ilm': [_*100 for _ in omp_ilm]
#     ,'grasp_flm_pre': [_*100 for _ in grasp_flm_pre]
#     ,'grasp_flm_after': [_*100 for _ in grasp_flm_after]
#     ,'grasp_ilm': [_*100 for _ in grasp_ilm]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))


# tricks_exp_flm_after_prune
# omp_flm_pre = [0.7401, 0.7415, 0.7391]
# omp_flm_after = [0.7401, 0.7378, 0.75]
# omp_ilm = [0.8051, 0.8308, 0.7939]
# grasp_flm_pre = [0.7401, 0.6009, 0.5669]
# grasp_flm_after = [0.7401, 0.5506, 0.522]
# grasp_ilm = [0.8051, 0.5622, 0.5446]


# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'flm pretrain&after_prune - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728][:3]]
#     ,'omp_flm_pre': [_*100 for _ in omp_flm_pre]
#     ,'omp_flm_after': [_*100 for _ in omp_flm_after]
#     ,'omp_ilm': [_*100 for _ in omp_ilm]
#     ,'grasp_flm_pre': [_*100 for _ in grasp_flm_pre]
#     ,'grasp_flm_after': [_*100 for _ in grasp_flm_after]
#     ,'grasp_ilm': [_*100 for _ in grasp_ilm]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))




# prompt_loc_exp
# pad = [0.6286, 0.6224]
# fix = [0.5754, 0.5814]
# random = [0.4755, 0.4678]

# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'VP Location - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728][:2]]
#     ,'pad': [_*100 for _ in pad]
#     ,'fix': [_*100 for _ in fix]
#     ,'random': [_*100 for _ in random]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))

# prompt_prune_exp
# imp_flm = [0.7401, 0.744, 0.738, 0.7435, 0.7239]
# imp_ilm = [0.8051, 0.8267, 0.813, 0.8159, 0.7851]
# omp_flm = [0.7401, 0.7415, 0.7391, 0.734, 0.7254]
# omp_ilm = [0.8051, 0.8308, 0.7939, 0.7929, 0.7799]
# grasp_flm = [0.7401, 0.6009, 0.5669, 0.5299, 0.2504]
# grasp_ilm = [0.8051, 0.5622, 0.5446, 0.5231, 0.4542]
# hydra_flm = [0.7401, 0.9491, 0.9395, 0.9157, 0.905]
# hydra_ilm = [0.8051, 0.9489, 0.9364, 0.9222, 0.9002]

# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'VP Prune - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728][:5]]
#     ,'IMP-FLM': [_*100 for _ in imp_flm]
#     ,'IMP-ILM': [_*100 for _ in imp_ilm]
#     ,'OMP-FLM': [_*100 for _ in omp_flm]
#     ,'OMP-ILM': [_*100 for _ in omp_ilm]
#     ,'GraSP-FLM': [_*100 for _ in grasp_flm]
#     ,'GraSP-ILM': [_*100 for _ in grasp_ilm]
#     ,'Hydra-FLM': [_*100 for _ in hydra_flm]
#     ,'Hydra-ILM': [_*100 for _ in hydra_ilm]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('Weight Density(%)')
# plt.ylabel('Accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))




# 0604
# choose the best input_size
# input_32 = [0.5522, 0.5565, 0.5599, 0.5636, 0.5581]
# input_64 = [0.6043, 0.6007, 0.6074, 0.6046, 0.5961]
# input_96 = [0.6374, 0.6385, 0.6403, 0.6382, 0.635]
# input_128 = [0.6492, 0.6542, 0.661, 0.6453, 0.6454]
# input_160 = [0.7401, 0.744, 0.738, 0.7435, 0.7239]
# input_192 = [0.6773, 0.6789, 0.68, 0.6758, 0.6689]
# input_224 = [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]

# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'Input size - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728][:5]]
#     ,'SIZE-32': [_*100 for _ in input_32]
#     ,'SIZE-64': [_*100 for _ in input_64]
#     ,'SIZE-96': [_*100 for _ in input_96]
#     ,'SIZE-128': [_*100 for _ in input_128]
#     ,'SIZE-160': [_*100 for _ in input_160]
#     ,'SIZE-192': [_*100 for _ in input_192]
#     ,'SIZE-224': [_*100 for _ in input_224]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))


# choose the best pad_size
# pad_112 = [0.6179, 0.6147, 0.6077, 0.6173, 0.6145]
# pad_96 = [0.6118, 0.6199, 0.6159, 0.6226, 0.6082]
# pad_80 = [0.6166, 0.6202, 0.6142, 0.6115, 0.6159]
# pad_64 = [0.6135, 0.6112, 0.6037, 0.6072, 0.61]
# pad_48 = [0.614, 0.6055, 0.6066, 0.6072, 0.5923]
# pad_32 = [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]
# pad_16 = [0.5896, 0.5918, 0.5919, 0.5892, 0.5845]


# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'Pad size - Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728][:5]]
#     ,'SIZE-16': [_*100 for _ in [0.5896, 0.5918, 0.5919, 0.5892, 0.5845]]
#     ,'SIZE-32': [_*100 for _ in [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]]
#     ,'SIZE-48': [_*100 for _ in [0.614, 0.6055, 0.6066, 0.6072, 0.5923]]
#     ,'SIZE-64': [_*100 for _ in [0.6135, 0.6112, 0.6037, 0.6072, 0.61]]
#     ,'SIZE-80': [_*100 for _ in [0.6166, 0.6202, 0.6142, 0.6115, 0.6159]]
#     ,'SIZE-96': [_*100 for _ in [0.6118, 0.6199, 0.6159, 0.6226, 0.6082]]
#     ,'SIZE-112': [_*100 for _ in [0.6179, 0.6147, 0.6077, 0.6173, 0.6145]]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))





# 0603

# prune without tune
# imp_flm = [0.5042, 0.4983, 0.4723, 0.4754, 0.4616, 0.455, 0.3472, 0.1568, 0.1268, 0.0971]
# imp_ilm = [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]
# omp_flm = [0.5042, 0.499, 0.475, 0.4825, 0.4797, 0.4214, 0.2975, 0.1296, 0.1028, 0.1003]
# omp_ilm = [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]
# grasp_flm = [0.5042, 0.0848, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# grasp_ilm = [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]
# hydra_flm = [0.5042, 0.1104, 0.1149, 0.0976, 0.1053, 0.1, 0.1, 0.1, 0.1, 0.1]
# hydra_ilm = [0.5042, 0.1724, 0.1217, 0.1355, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'Prune without tune Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'IMP-FLM': [_*100 for _ in [0.5042, 0.4983, 0.4723, 0.4754, 0.4616, 0.455, 0.3472, 0.1568, 0.1268, 0.0971]]
#     ,'IMP-ILM': [_*100 for _ in [0.5042, 0.4985, 0.4866, 0.4923, 0.4162, 0.2663, 0.2383, 0.1745, 0.1874, 0.1947]]
#     ,'OMP-FLM': [_*100 for _ in [0.5042, 0.499, 0.475, 0.4825, 0.4797, 0.4214, 0.2975, 0.1296, 0.1028, 0.1003]]
#     ,'OMP-ILM': [_*100 for _ in [0.5042, 0.4996, 0.4892, 0.4718, 0.4152, 0.2738, 0.1921, 0.1938, 0.2072, 0.1586]]
#     ,'GraSP-FLM': [_*100 for _ in [0.5042, 0.0848, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
#     ,'GraSP-ILM': [_*100 for _ in [0.5042, 0.1647, 0.1927, 0.1023, 0.1001, 0.1114, 0.1, 0.1, 0.1, 0.1]]
#     # ,'Hydra-FLM': [_*100 for _ in [0.5042, 0.1104, 0.1149, 0.0976, 0.1053, 0.1, 0.1, 0.1, 0.1, 0.1]]
#     # ,'Hydra-ILM': [_*100 for _ in [0.5042, 0.1724, 0.1217, 0.1355, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))






# prune with finetune
# imp_ff_flm = [0.9625, 0.9668, 0.9652, 0.9656, 0.9676, 0.9668, 0.9651, 0.9672, 0.9649, 0.9648]
# imp_ff_ilm = [0.9625, 0.9666, 0.9682, 0.9655, 0.966, 0.9661, 0.9645, 0.9638, 0.9652, 0.962]
# omp_ff_flm = [0.9625, 0.9633, 0.9627, 0.9615, 0.9604, 0.9586, 0.9581, 0.9582, 0.9563, 0.9552]
# omp_ff_ilm = [0.9625, 0.9632, 0.9635, 0.9621, 0.9631, 0.9586, 0.9592, 0.9582, 0.9541, 0.9529]
# grasp_ff_flm = [0.9625, 0.9602, 0.9585, 0.9532, 0.9526, 0.9451, 0.9406, 0.9383, 0.9386, 0.9322]
# grasp_ff_ilm = [0.9625, 0.9599, 0.9544, 0.9511, 0.9483, 0.9453, 0.944, 0.9399, 0.9375, 0.9321]
# hydra_ff_flm = [0.9625, 0.9628, 0.9592, 0.9544, 0.9475, 0.9383, 0.9276, 0.9246, 0.9183, 0.9201]
# hydra_ff_ilm = [0.9625, 0.9636, 0.9579, 0.9564, 0.9449, 0.9363, 0.9288, 0.9258, 0.923, 0.9193]

# ff grasp/hydra
# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'Finetune - GraSP+Hydra Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'GraSP-FLM': [_*100 for _ in [0.9625, 0.9602, 0.9585, 0.9532, 0.9526, 0.9451, 0.9406, 0.9383, 0.9386, 0.9322]]
#     ,'GraSP-ILM': [_*100 for _ in [.9625, 0.9599, 0.9544, 0.9511, 0.9483, 0.9453, 0.944, 0.9399, 0.9375, 0.9321]]
#     ,'Hydra-FLM': [_*100 for _ in [0.9625, 0.9628, 0.9592, 0.9544, 0.9475, 0.9383, 0.9276, 0.9246, 0.9183, 0.9201]]
#     ,'Hydra-ILM': [_*100 for _ in [0.9625, 0.9636, 0.9579, 0.9564, 0.9449, 0.9363, 0.9288, 0.9258, 0.923, 0.9193]]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))



# ff imp/omp/grasp/hydra
# save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
# image_name = 'Finetune - Prune Accuracy'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'IMP-FLM': [_*100 for _ in [0.9625, 0.9668, 0.9652, 0.9656, 0.9676, 0.9668, 0.9651, 0.9672, 0.9649, 0.9648]]
#     ,'IMP-ILM': [_*100 for _ in [0.9625, 0.9666, 0.9682, 0.9655, 0.966, 0.9661, 0.9645, 0.9638, 0.9652, 0.962]]
#     ,'OMP-FLM': [_*100 for _ in [0.9625, 0.9633, 0.9627, 0.9615, 0.9604, 0.9586, 0.9581, 0.9582, 0.9563, 0.9552]]
#     ,'OMP-ILM': [_*100 for _ in [0.9625, 0.9632, 0.9635, 0.9621, 0.9631, 0.9586, 0.9592, 0.9582, 0.9541, 0.9529]]
#     ,'GraSP-FLM': [_*100 for _ in [0.9625, 0.9602, 0.9585, 0.9532, 0.9526, 0.9451, 0.9406, 0.9383, 0.9386, 0.9322]]
#     ,'GraSP-ILM': [_*100 for _ in [.9625, 0.9599, 0.9544, 0.9511, 0.9483, 0.9453, 0.944, 0.9399, 0.9375, 0.9321]]
#     ,'Hydra-FLM': [_*100 for _ in [0.9625, 0.9628, 0.9592, 0.9544, 0.9475, 0.9383, 0.9276, 0.9246, 0.9183, 0.9201]]
#     ,'Hydra-ILM': [_*100 for _ in [0.9625, 0.9636, 0.9579, 0.9564, 0.9449, 0.9363, 0.9288, 0.9258, 0.923, 0.9193]]
# }
# for k in results.keys():
#     if k != 'density':
#         plt.plot(results['density'], results[k], label=k)

# plt.title(image_name)
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))



# image resize
# save_dir = 'results/Prune_LM_VP'
# image_name = 'image resize'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'resize': [32, 128, 200]
#     ,'VP-FLM': [65.28,  69.58, 57.93]
# }
# plt.plot(results['resize'], results['VP-FLM'])
# plt.title('VP-FLM-Accuracy')
# plt.xlabel('resize')
# plt.ylabel('accuracy(%)')
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))





# grasp_hydra
# save_dir = 'results/Prune_LM_VP'
# image_name = 'grasp_hydra'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'Grasp-FLM': [15.83,  10.00,  12.24,  13.29,  11.20,  10.00,  13.74,  10.00,  10.00]
#     ,'Grasp-ILM': [12.09,  10.00,  11.77,  12.07,  9.80,   10.00,  14.19,  10.00,  10.00]
#     ,'Hydra-FLM': [10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00]
#     ,'Hydra-ILM': [10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00]
# }
# increase_density = [100, 95, 90] + [round(100*_,2) for _ in [ 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
# grasp_flm = [65.64, 54.16, 19.46] + [15.83,  10.00,  12.24,  13.29,  11.20,  10.00,  13.74,  10.00,  10.00]
# hydra_flm = [65.64, 76.67, 74.20] + [10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00]
# plt.plot(increase_density, grasp_flm, label='Grasp-FLM')
# plt.plot(results['density'], results['Grasp-ILM'], label='Grasp-ILM')
# plt.plot(increase_density, hydra_flm, label='Hydra-FLM')
# plt.plot(results['density'], results['Hydra-ILM'], label='Hydra-ILM')
# plt.title('GraSP/Hydra + VP')
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))




# imp_omp
# save_dir = 'results/Prune_LM_VP'
# image_name = 'imp_omp'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'IMP-FLM': [65.64, 61.81,  61.70,  62.18,  61.46,  61.23,  60.03,  58.42,  56.38,  53.61]
#     ,'IMP-ILM': [65.64, 61.64,  61.44,  61.51,  61.31,  60.08,  58.78,  57.05,  54.75,  51.83]
#     ,'OMP-FLM': [65.64, 61.60,  62.10,  61.32,  61.79,  61.21,  60.12,  57.10,  53.51,  49.76]
#     ,'OMP-ILM': [65.64, 62.65,  62.61,  61.57,  60.67,  59.25,  56.72,  54.90,  50.24,  50.11]
# }

# plt.plot(results['density'], results['IMP-FLM'], label='IMP-FLM')
# plt.plot(results['density'], results['IMP-ILM'], label='IMP-ILM')
# plt.plot(results['density'], results['OMP-FLM'], label='OMP-FLM')
# plt.plot(results['density'], results['OMP-ILM'], label='OMP-ILM')

# plt.title('IMP/OMP + VP')
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))







# imp_omp_grasp_hydra
# save_dir = 'results/Prune_LM_VP'
# image_name = 'imp_omp_grasp_hydra'
# os.makedirs(save_dir, exist_ok=True)
# results = {
#     'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
#     ,'IMP-FLM': [65.64, 61.81,  61.70,  62.18,  61.46,  61.23,  60.03,  58.42,  56.38,  53.61]
#     ,'IMP-ILM': [65.64, 61.64,  61.44,  61.51,  61.31,  60.08,  58.78,  57.05,  54.75,  51.83]
#     ,'OMP-FLM': [65.64, 61.60,  62.10,  61.32,  61.79,  61.21,  60.12,  57.10,  53.51,  49.76]
#     ,'OMP-ILM': [65.64, 62.65,  62.61,  61.57,  60.67,  59.25,  56.72,  54.90,  50.24,  50.11]
#     ,'Grasp-FLM': [65.64, 15.83,  10.00,  12.24,  13.29,  11.20,  10.00,  13.74,  10.00,  10.00]
#     ,'Grasp-ILM': [65.64, 12.09,  10.00,  11.77,  12.07,  9.80,   10.00,  14.19,  10.00,  10.00]
#     ,'Hydra-FLM': [65.64, 10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00]
#     ,'Hydra-ILM': [65.64, 10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00]
# }
# plt.plot(results['density'], results['IMP-FLM'], label='IMP-FLM')
# plt.plot(results['density'], results['IMP-ILM'], label='IMP-ILM')
# plt.plot(results['density'], results['OMP-FLM'], label='OMP-FLM')
# plt.plot(results['density'], results['OMP-ILM'], label='OMP-ILM')
# plt.plot(results['density'], results['Grasp-FLM'], label='Grasp-FLM')
# plt.plot(results['density'], results['Grasp-ILM'], label='Grasp-ILM')
# plt.plot(results['density'], results['Hydra-FLM'], label='Hydra-FLM')
# plt.plot(results['density'], results['Hydra-ILM'], label='Hydra-ILM')
# plt.title('IMP/OMP/GraSP/Hydra + VP')
# plt.xlabel('weight density(%)')
# plt.ylabel('accuracy(%)')
# plt.legend(loc=2)
# plt.grid()
# plt.savefig(os.path.join(save_dir, image_name+'.png'))
# plt.close()
# result_df = pd.DataFrame(results)
# result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))



# 0522
# ,'IMP-FLM': [46.88,95.56,95.71,95.64,95.6,95.65,95.54,95.7,95.57,95.55]
# ,'IMP-ILM': [46.88,91.13,94.67,93.18,94.26,94.97,95.53,95.24,95.36,95.42]
# ,'OMP-FLM': [46.88,95.67,95.52,95.86,95.52,95.87,95.95,95.63,95.50,95.73]
# ,'OMP-ILM': [46.88,95.33,95.70,95.83,95.74,95.75,95.72,95.78,95.74,95.78]
# ,'Grasp-FLM': [46.88,92.70,93.14,93.34,85.30,85.24,85.31,85.10,84.84,84.78]
# ,'Grasp-ILM': [46.88,93.06,92.83,93.05,94.63,94.38,93.99,93.85,93.57,93.56]
# ,'IMP-FLM': [46.88,44.64,41.92,38.66,30.84,24.54,13.05,10.15,10,10]
# ,'IMP-ILM': [46.88,44.42,41.06,37.29,32.26,26.35,19.57,17.62,14.81,18.7]
# ,'OMP-FLM': [46.88,44.41,41.71,38.67,29.38,20.32,11.19,10.00,10.00,10.00]
# ,'OMP-ILM': [46.88,44.65,40.66,37.30,31.58,24.42,18.32,15.20,16.40,16.50]
# ,'Grasp-FLM': [46.88,10.22,10.00,9.83,12.06,10.00,10.00,10.00,10.00,10.00]
# ,'Grasp-ILM': [46.88,15.24,13.31,16.32,11.83,10.00,11.13,10.00,10.00,10.00]