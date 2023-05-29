import matplotlib.pyplot as plt
import pandas as pd

results = {
    'density': [round(100*_,2) for _ in [1, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144, 0.2097152, 0.16777216, 0.134217728]]
    ,'IMP-FLM': [46.88,95.56,95.71,95.64,95.6,95.65,95.54,95.7,95.57,95.55]
    ,'IMP-ILM': [46.88,91.13,94.67,93.18,94.26,94.97,95.53,95.24,95.36,95.42]
    ,'OMP-FLM': [46.88,95.67,95.52,95.86,95.52,95.87,95.95,95.63,95.50,95.73]
    ,'OMP-ILM': [46.88,95.33,95.70,95.83,95.74,95.75,95.72,95.78,95.74,95.78]
    ,'Grasp-FLM': [46.88,92.70,93.14,93.34,85.30,85.24,85.31,85.10,84.84,84.78]
    ,'Grasp-ILM': [46.88,93.06,92.83,93.05,94.63,94.38,93.99,93.85,93.57,93.56]
    # ,'IMP-FLM': [46.88,44.64,41.92,38.66,30.84,24.54,13.05,10.15,10,10]
    # ,'IMP-ILM': [46.88,44.42,41.06,37.29,32.26,26.35,19.57,17.62,14.81,18.7]
    # ,'OMP-FLM': [46.88,44.41,41.71,38.67,29.38,20.32,11.19,10.00,10.00,10.00]
    # ,'OMP-ILM': [46.88,44.65,40.66,37.30,31.58,24.42,18.32,15.20,16.40,16.50]
    # ,'Grasp-FLM': [46.88,10.22,10.00,9.83,12.06,10.00,10.00,10.00,10.00,10.00]
    # ,'Grasp-ILM': [46.88,15.24,13.31,16.32,11.83,10.00,11.13,10.00,10.00,10.00]
}
plt.plot(results['density'], results['IMP-FLM'], label='IMP-FLM')
plt.plot(results['density'], results['IMP-ILM'], label='IMP-ILM')
plt.plot(results['density'], results['OMP-FLM'], label='OMP-FLM')
plt.plot(results['density'], results['OMP-ILM'], label='OMP-ILM')
plt.plot(results['density'], results['Grasp-FLM'], label='Grasp-FLM')
plt.plot(results['density'], results['Grasp-ILM'], label='Grasp-ILM')
plt.title('IMP/OMP/GraSP + Finetune')
plt.xlabel('weight density(%)')
plt.ylabel('accuracy(%)')
plt.legend(loc=2)
plt.savefig('results/0522/Finetune.png')
plt.close()
result_df = pd.DataFrame(results)
result_df.to_csv('results/0522/Finetune.csv')
