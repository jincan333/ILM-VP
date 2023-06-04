import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

# 0604
# choose the best input_size
input_32 = [0.5522, 0.5565, 0.5599, 0.5636, 0.5581]
input_64 = [0.6043, 0.6007, 0.6074, 0.6046, 0.5961]
input_96 = [0.6374, 0.6385, 0.6403, 0.6382, 0.635]
input_128 = [0.6492, 0.6542, 0.661, 0.6453, 0.6454]
input_160 = [0.7401, 0.744, 0.738, 0.7435, 0.7239]
input_192 = [0.6773, 0.6789, 0.68, 0.6758, 0.6689]
input_224 = [0.6286, 0.6224, 0.6128, 0.6162, 0.5988]

save_dir = os.path.join('results/Prune_LM_VP', str(datetime.datetime.now().date()))
image_name = 'Input size - Accuracy'
os.makedirs(save_dir, exist_ok=True)
results = {
    'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728][:5]]
    ,'SIZE-32': [_*100 for _ in input_32]
    ,'SIZE-64': [_*100 for _ in input_64]
    ,'SIZE-96': [_*100 for _ in input_96]
    ,'SIZE-128': [_*100 for _ in input_128]
    ,'SIZE-160': [_*100 for _ in input_160]
    ,'SIZE-192': [_*100 for _ in input_192]
    ,'SIZE-224': [_*100 for _ in input_224]
}
for k in results.keys():
    if k != 'density':
        plt.plot(results['density'], results[k], label=k)

plt.title(image_name)
plt.xlabel('weight density(%)')
plt.ylabel('accuracy(%)')
plt.legend(loc=2)
plt.grid()
plt.savefig(os.path.join(save_dir, image_name+'.png'))
plt.close()
result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(save_dir, image_name+'.csv'))


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
#     ,'Hydra-FLM': [_*100 for _ in [0.5042, 0.1104, 0.1149, 0.0976, 0.1053, 0.1, 0.1, 0.1, 0.1, 0.1]]
#     ,'Hydra-ILM': [_*100 for _ in [0.5042, 0.1724, 0.1217, 0.1355, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
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