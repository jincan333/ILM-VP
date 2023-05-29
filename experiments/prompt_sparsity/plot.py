import matplotlib.pyplot as plt
import pandas as pd
import os

# image resize prune
save_dir = 'results/Prune_LM_VP'
image_name = 'VP Size - IMP Prune Accuracy'
os.makedirs(save_dir, exist_ok=True)
results = {
    'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
    ,'SIZE-32': [65.64, 61.81,  61.70,  62.18,  61.46,  61.23,  60.03,  58.42,  56.38,  53.61]
    ,'SIZE-128': [69.58,  64.85,  64.74,  64.08, 61.50, 60.42]
    ,'SIZE-200': [57.93,  51.95,  50.21,  50.59,  47.85,  45.77,  41.12,  36.69,  30.36,  23.85]
}

plt.plot(results['density'], results['SIZE-32'], label='SIZE-32')
plt.plot(results['density'][:6], results['SIZE-128'], label='SIZE-128')
plt.plot(results['density'], results['SIZE-200'], label='SIZE-200')

plt.title('VP size influence accuracy')
plt.xlabel('weight density(%)')
plt.ylabel('accuracy(%)')
plt.legend(loc=2)
plt.grid()
plt.savefig(os.path.join(save_dir, image_name+'.png'))
plt.close()
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