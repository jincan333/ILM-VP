import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

def winning_ticket_gap(unpruned_acc, avg_acc_list, num_sparsities):
    x = np.arange(num_sparsities)
    x_density = 1-  0.8 ** x
    print("winning ticket acc gap:",np.max(avg_acc_list) - unpruned_acc)
    print("Achived at the sparsity of {}".format(x_density[np.argmax(avg_acc_list)]))
    return 0 

def last_winning_ticket_sparsity(unpruned_acc, avg_acc_list, num_sparsities, err_list):
    x = np.arange(num_sparsities)
    x_density = 1-  0.8 ** x
    print("last winning ticket sparsity:", x_density[avg_acc_list - unpruned_acc >= 0])
    print("Accuracy:", avg_acc_list[avg_acc_list - unpruned_acc >= 0] )
    print("Std:", err_list[avg_acc_list - unpruned_acc >= 0] )
    return 0

def extract_y_err(y_dense, performance_str):
    if '(' in performance_str or ')' in performance_str:
        y = re.findall('(\d+\.\d+)(?=\()', performance_str)
        err = re.findall('(?<=\()(\d+\.\d+)', performance_str)
        y = [float(num) for num in y]
        err = [float(num) for num in err]       
    else:
        y = [float(i) for i in performance_str.split()]
        err = [0] * len(y)

    y.insert(0, y_dense)
    err.insert(0, 0)
    return np.array(y), np.array(err)

if __name__ == "__main__":
    # num = 14
    # # x_grid = np.array(range(num))
    # step = 1
    # index = np.arange(0, num, step)
    # x = np.arange(num)[index]
    # x_density = 100 - 100 * (0.8 ** x)
    # x_grid = x_density
    # x_density_list = ['{:.2f}'.format(value) for value in x_density]

    # num = 14
    # x_grid = np.array(range(num))
    # step = 1
    # index = np.arange(0, num, step)
    # x = np.arange(num)[index]
    # x_density = 100 - 100 * (0.8 ** x)
    # x_grid = x_density
    # x_density_list = ['{:.2f}'.format(value) for value in x_density]
    
    # 7, 11; 9, 20
    title = 'Steps'
    num, imp_num = 20, 20
    y_min, y_max = 100,1800
    x_label = "Sparsity (%)"
    y_label = "Steps"
    
    # 10, 20
    x_sparsity_list = np.array([40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_IMP_sparsity_list = np.array([20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])
    x_grid = x_IMP_sparsity_list

    # y_dense_time = 118
    # y_VPNs_time = np.array([110 for i in range(num - 1)])
    # y_BiP_time = np.array([145 for i in range(num - 1)])
    # y_HYDRA_time = np.array([136 for i in range(num - 1)])
    # y_IMP_time = np.array([115.2 * i for i in range(1, num)])
    # y_OMP_time = np.array([115.2 for i in range(num - 1)])
    # y_GraSP_time = np.array([120 for i in range(num - 1)])

    # y_dense_epochs = 120
    # y_VPNs_epochs = np.array([30 for i in range(num - 1)])
    # y_OMP_epochs = np.array([120 for i in range(num - 1)])
    # y_IMP_epochs = np.array([120 * i for i in range(1, num)])
    # y_HYDRA_epochs = np.array([60 for i in range(num - 1)])

    y_dense_steps = 120
    y_VPNs_steps = np.array([120 for i in range(num - 1)])
    y_IMP_steps = np.array([120 * i for i in range(1, num)]) 


    # Canvas setting
    width = 14
    height = 12
    plt.figure(figsize=(width, height))

    sns.set_theme()
    plt.grid(visible=True, which='major', linestyle='-', linewidth=4)
    plt.grid(visible=True, which='minor')
    plt.minorticks_on()
    plt.rcParams['font.serif'] = 'Times New Roman'

    markersize = 20
    linewidth = 2
    markevery = 1
    fontsize = 40
    alpha = 1

    # Color Palette
    best_color = 'green'
    best_alpha = 1.0
    dense_color = 'black'
    dense_alpha = 1.0
    SynFlow_color = 'red'
    SynFlow_alpha = 0.9
    VPNs_color = 'green'
    VPNs_alpha = 0.9
    BiP_color = 'hotpink'
    BiP_alpha = alpha
    HYDRA_color = 'blue'
    HYDRA_alpha = alpha - 0.1
    IMP_color = 'darkorange'
    IMP_alpha = alpha
    OMP_color = 'darkolivegreen'
    OMP_alpha = alpha
    GraSP_color = 'purple'
    GraSP_alpha = alpha
    SNIP_color = 'gold'
    SNIP_alpha = alpha
    Random_color = 'violet'
    Random_alpha = alpha


    fill_in_alpha = 0.2

    # l_dense = plt.axhline(y=y_dense, color=dense_color, linestyle='--', linewidth=3, label="Dense")

    # l_HYDRA = plt.plot(x_grid, y_HYDRA, color=HYDRA_color, marker='v', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Hydra", alpha=HYDRA_alpha)
    # plt.fill_between(x_grid, y_HYDRA - y_HYDRA_err, y_HYDRA + y_HYDRA_err, color=HYDRA_color, alpha=fill_in_alpha)

    # l_IMP = plt.plot(x_IMP_sparsity_list, y_IMP, color=IMP_color, marker='o', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize, label="IMP", alpha=IMP_alpha)
    # plt.fill_between(x_IMP_sparsity_list, y_IMP - y_IMP_err, y_IMP + y_IMP_err, color=IMP_color, alpha=fill_in_alpha)

    # l_OMP = plt.plot(x_grid, y_OMP, color=OMP_color, marker='s', markevery=markevery, linestyle='-', linewidth=linewidth,
    #                 markersize=markersize, label="OMP", alpha=OMP_alpha)
    # plt.fill_between(x_grid, y_OMP - y_OMP_err, y_OMP + y_OMP_err, color=OMP_color, alpha=fill_in_alpha)

    # l_GraSP = plt.plot(x_grid, y_GraSP, color=Grasp_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Grasp", alpha=Grasp_alpha)
    # plt.fill_between(x_grid, y_GraSP - y_GraSP_err, y_GraSP + y_GraSP_err, color=Grasp_color, alpha=fill_in_alpha)

    # l_BiP = plt.plot(x_grid, y_BiP, color=BiP_color, marker='v', markevery=markevery, linestyle='-',
    #                 linewidth=linewidth,
    #                 markersize=markersize + 4, label="BiP", alpha=BiP_alpha)
    # plt.fill_between(x_grid, y_BiP - y_BiP_err, y_BiP + y_BiP_err, color=BiP_color, alpha=fill_in_alpha)

    # l_SynFlow = plt.plot(x_grid, y_SynFlow, color=SynFlow_color, marker='v', markevery=markevery, linestyle='-',
    #             linewidth=linewidth,
    #             markersize=markersize + 4, label="SynFlow", alpha=SynFlow_alpha)
    # plt.fill_between(x_grid, y_SynFlow - y_SynFlow_err, y_SynFlow + y_SynFlow_err, color=SynFlow_color, alpha=fill_in_alpha)

    # l_SNIP = plt.plot(x_grid, y_SNIP, color=SNIP_color, marker='o', markevery=markevery, linestyle='-',
    #             linewidth=linewidth,
    #             markersize=markersize + 4, label="SNIP", alpha=SNIP_alpha)
    # plt.fill_between(x_grid, y_SNIP - y_SNIP_err, y_SNIP + y_SNIP_err, color=SNIP_color, alpha=fill_in_alpha)

    # l_Random = plt.plot(x_grid, y_Random, color=Random_color, marker='o', markevery=markevery, linestyle='-',
    #         linewidth=linewidth,
    #         markersize=markersize + 4, label="Random", alpha=Random_alpha)
    # plt.fill_between(x_grid, y_Random - y_Random_err, y_Random + y_Random_err, color=Random_color, alpha=fill_in_alpha)

    # l_VPNs = plt.plot(x_grid, y_VPNs, color=VPNs_color, marker='*', markevery=markevery, linestyle='-',
    #     linewidth=linewidth,
    #     markersize=markersize + 4, label="VPNs", alpha=VPNs_alpha)
    # plt.fill_between(x_grid, y_VPNs - y_VPNs_err, y_VPNs + y_VPNs_err, color=VPNs_color, alpha=fill_in_alpha)

    # plt.ylim([y_min, y_max])
    # plt.xlim(0, 100)

    # plt.legend(fontsize=fontsize - 15, loc=3, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    # plt.xlabel(x_label, fontsize=fontsize)
    # plt.ylabel(y_label, fontsize=fontsize)
    # plt.xticks(x_grid, x_sparsity_list, rotation=0, fontsize=fontsize)
    # plt.xscale("linear")
    # plt.yticks(fontsize=fontsize)

    # plt.title(title, fontsize=fontsize)
    # plt.tight_layout()
    # plt.twinx()
    y_time_label = y_label
    linewidth = 2
    # l_dense = plt.axhline(y=y_dense_time, color=dense_color, linestyle='--', linewidth=3, label="Dense")
    # time_linestyle = '-'
    # t_IMP = plt.plot(x_grid, y_IMP_time, color=IMP_color, marker='o', markersize=markersize, alpha=IMP_alpha, label="IMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    # t_BiP = plt.plot(x_grid, y_BiP_time, color=BiP_color, marker='o', markersize=markersize, alpha=BiP_alpha, label="BiP", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # t_VPNs = plt.plot(x_grid, y_VPNs_time, color=VPNs_color, marker='o', markersize=markersize, alpha=VPNs_alpha, label="VPNs", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # t_HYDRA = plt.plot(x_grid, y_HYDRA_time, color=HYDRA_color, marker='o', markersize=markersize, alpha=HYDRA_alpha, label="HYDRA",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # t_GraSP = plt.plot(x_grid, y_GraSP_time, color=GraSP_color, marker='o', markersize=markersize, alpha=GraSP_alpha, label="GraSP",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # t_OMP = plt.plot(x_grid, y_OMP_time, color=OMP_color, marker='o', markersize=markersize, alpha=OMP_alpha + 0., label="OMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)

    # l_dense = plt.axhline(y=y_dense_epochs, color=dense_color, linestyle='--', linewidth=3, label="Dense")
    # time_linestyle = '-'
    # t_IMP = plt.plot(x_grid, y_IMP_epochs, color=IMP_color, marker='o', markersize=markersize, alpha=IMP_alpha, label="IMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    # t_VPNs = plt.plot(x_grid, y_VPNs_epochs, color=VPNs_color, marker='o', markersize=markersize, alpha=VPNs_alpha, label="VPNs", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # t_HYDRA = plt.plot(x_grid, y_HYDRA_epochs, color=HYDRA_color, marker='o', markersize=markersize, alpha=HYDRA_alpha, label="HYDRA",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # t_OMP = plt.plot(x_grid, y_OMP_epochs, color=OMP_color, marker='o', markersize=markersize, alpha=OMP_alpha + 0., label="OMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)

    l_dense = plt.axhline(y=y_dense_steps, color=dense_color, linestyle='--', linewidth=3, label="Dense")
    time_linestyle = '-'
    t_IMP = plt.plot(x_grid, y_IMP_steps, color=IMP_color, marker='o', markersize=markersize, alpha=IMP_alpha, label="IMP", linestyle=time_linestyle,
                    linewidth=linewidth)
    t_VPNs = plt.plot(x_grid, y_VPNs_steps, color=VPNs_color, marker='o', markersize=markersize, alpha=VPNs_alpha, label="VPNs", linestyle=time_linestyle,
                     linewidth=linewidth)
    
    plt.legend(fontsize=fontsize - 10, loc=2, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.yscale('log')
    plt.xlabel(x_label, fontsize=fontsize-10)
    plt.ylabel(y_time_label, fontsize=fontsize-10)
    plt.xticks([20, 40, 60, 80, 100], rotation=0, fontsize=fontsize-15)
    plt.yticks(fontsize=fontsize-15)
    plt.ylim(y_min, y_max)
    plt.xlim(10, 100)
    plt.savefig(f"pic/computation_cost/{title}.pdf")
    plt.show()
    plt.close()










