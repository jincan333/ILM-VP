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

    # y_PFTT_time = np.insert(np.array([180 for i in range(num - 1)]), 0, 0)
    # y_BiP_time = np.insert(np.array([225 for i in range(num - 1)]), 0, 0)
    # y_hydra_time = np.insert(np.array([136 for i in range(num - 1)]), 0, 0)
    # y_IMP_time = np.array([115.2 * i for i in range(num)])
    # y_OMP_time = np.insert(np.array([115.2 for i in range(num - 1)]), 0, 0)
    # y_Grasp_time = np.insert(np.array([120 for i in range(num - 1)]), 0, 0)
    
    # 7, 11; 9, 20
    title = 'Prompt Size'
    num, imp_num = 7, 11
    y_dense = 82.10
    y_min, y_max = 80.5,83.5
    
    # 10, 20
    x_sparsity_list = np.array([0, 40, 50, 60, 70, 80, 90, 95, 99][:num])
    x_grid = x_sparsity_list
    x_IMP_sparsity_list = np.array([0, 20.00, 36.00, 48.80, 59.00, 67.20, 73.80, 79.03, 83.22, 86.58, 89.26, 91.41, 93.13, 94.50, 95.60, 96.50, 97.75, 98.20, 98.56, 98.85][:imp_num])
    
    # y_IMP = np.array([y_dense,73.01,72.72,72.36,71.97,71.18,70.49,69.52,68.43,67.17,65.89])
    # y_IMP_err = np.array([0,0.10,0.16,0.10,0.17,0.39,0.21,0.25,0.39,0.33,0.18])

    prompt16   ='83.47 	83.29 	83.29 	82.98 	82.17 	80.87 '
    prompt32   ='83.22 	83.41 	83.26 	82.98 	81.79 	80.87 '
    prompt48   ='83.06 	83.17 	83.13 	83.08 	82.02 	80.39 '
    prompt64   ='83.00 	83.19 	83.20 	83.09 	82.09 	80.45 '

    input128   ='80.24 	80.41 	81.01 	80.27 	79.99 	78.48 '
    input160   ='81.87 	82.82 	82.22 	81.79 	80.80 	79.80 '
    input192   ='82.73 	83.13 	82.80 	82.61 	82.24 	80.57 '
    input224   ='83.47 	83.29 	83.29 	82.98 	82.17 	80.87 '

    pad    = '83.47 	83.29 	83.29 	82.98 	82.17 	80.87 '
    fix    = '82.98 	83.08 	83.00 	82.93 	81.93 	80.55 '
    random = '82.93 	83.13 	82.98 	83.30 	82.22 	81.07 '

    y_prompt16, y_prompt16_err = extract_y_err(y_dense, prompt16)
    y_prompt32, y_prompt32_err = extract_y_err(y_dense, prompt32)
    y_prompt48, y_prompt48_err = extract_y_err(y_dense, prompt48)
    y_prompt64, y_prompt64_err = extract_y_err(y_dense, prompt64)

    y_input128, y_input128_err = extract_y_err(y_dense, input128)
    y_input160, y_input160_err = extract_y_err(y_dense, input160)
    y_input192, y_input192_err = extract_y_err(y_dense, input192)
    y_input224, y_input224_err = extract_y_err(y_dense, input224)

    y_pad, y_pad_err = extract_y_err(y_dense, pad)
    y_fix, y_fix_err = extract_y_err(y_dense, fix)
    y_random, y_random_err = extract_y_err(y_dense, random)

    y_best = np.max(y_pad)

    # print("IMP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_IMP, num)
    # print("Grasp winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_Grasp, num)
    # print("OMP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_OMP, num)
    # print("Hydra winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_hydra, num)
    # print("BiP winning ticket gap:")
    # winning_ticket_gap(y_IMP[0], y_BiP, num)

    # print("last IMP winning ticket:")
    # last_winning_ticket_sparsity(y_IMP[0], y_IMP, num, y_IMP_err)
    # print("last Grasp winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_Grasp, num, y_grasp_err)
    # print("last OMP winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_OMP, num, y_OMP_err)
    # print("last Hydra winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_hydra, num, y_hydra_err)
    # print("last BiP winning ticket gap:")
    # last_winning_ticket_sparsity(y_IMP[0], y_BiP, num, y_BiP_err)


    x_label = "Sparsity (%)"
    y_label = "Test Accuracy (%)"

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
    fontsize = 50
    alpha = 0.7

    # Color Palette
    best_color = 'green'
    best_alpha = 1.0
    dense_color = 'black'
    dense_alpha = 1.0
    SynFlow_color = 'red'
    SynFlow_alpha = 0.9
    prompt16_color = 'green'
    prompt16_alpha = 0.9
    prompt32_color = 'hotpink'
    prompt32_alpha = alpha
    prompt48_color = 'blue'
    prompt48_alpha = alpha - 0.1
    prompt64_color = 'darkorange'
    prompt64_alpha = alpha

    input224_color = 'green'
    input224_alpha = 0.9
    input160_color = 'hotpink'
    input160_alpha = alpha
    input192_color = 'blue'
    input192_alpha = alpha - 0.1
    input128_color = 'darkorange'
    input128_alpha = alpha

    pad_color = 'green'
    pad_alpha = 0.9
    fix_color = 'hotpink'
    fix_alpha = alpha
    random_color = 'blue'
    random_alpha = alpha - 0.1
    OMP_color = 'darkolivegreen'
    OMP_alpha = alpha
    Grasp_color = 'purple'
    Grasp_alpha = alpha
    SNIP_color = 'gold'
    SNIP_alpha = alpha
    Random_color = 'violet'
    Random_alpha = alpha


    fill_in_alpha = 0.2

    # plt.rcParams['font.sans-serif'] = 'Times New Roman'
    # plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # plt.minorticks_on()
    # plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


    l_dense = plt.axhline(y=y_dense, color=dense_color, linestyle='--', linewidth=3, label="Dense")

    l_prompt16 = plt.plot(x_grid, y_prompt16, color=prompt16_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Prompt Size 16", alpha=prompt16_alpha)
    plt.fill_between(x_grid, y_prompt16 - y_prompt16_err, y_prompt16 + y_prompt16_err, color=prompt16_color, alpha=fill_in_alpha)

    l_prompt32 = plt.plot(x_grid, y_prompt32, color=prompt32_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Prompt Size 32", alpha=prompt32_alpha)
    plt.fill_between(x_grid, y_prompt32 - y_prompt32_err, y_prompt32 + y_prompt32_err, color=prompt32_color, alpha=fill_in_alpha)

    l_prompt48 = plt.plot(x_grid, y_prompt48, color=prompt48_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Prompt Size 48", alpha=prompt48_alpha)
    plt.fill_between(x_grid, y_prompt48 - y_prompt48_err, y_prompt48 + y_prompt48_err, color=prompt48_color, alpha=fill_in_alpha)

    l_prompt64 = plt.plot(x_grid, y_prompt64, color=prompt64_color, marker='o', markevery=markevery, linestyle='-',
                      linewidth=linewidth,
                      markersize=markersize, label="Prompt Size 64", alpha=prompt64_alpha)
    plt.fill_between(x_grid, y_prompt64 - y_prompt64_err, y_prompt64 + y_prompt64_err, color=prompt64_color, alpha=fill_in_alpha)


    # l_input128 = plt.plot(x_grid, y_input128, color=input128_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Input Szie 128", alpha=input128_alpha)
    # plt.fill_between(x_grid, y_input128 - y_input128_err, y_input128 + y_input128_err, color=input128_color, alpha=fill_in_alpha)

    # l_input160 = plt.plot(x_grid, y_input160, color=input160_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Input Szie 160", alpha=input160_alpha)
    # plt.fill_between(x_grid, y_input160 - y_input160_err, y_input160 + y_input160_err, color=input160_color, alpha=fill_in_alpha)

    # l_input192 = plt.plot(x_grid, y_input192, color=input192_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Input Szie 192", alpha=input192_alpha)
    # plt.fill_between(x_grid, y_input192 - y_input192_err, y_input192 + y_input192_err, color=input192_color, alpha=fill_in_alpha)

    # l_input224 = plt.plot(x_grid, y_input224, color=input224_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Input Size 224", alpha=input224_alpha)
    # plt.fill_between(x_grid, y_input224 - y_input224_err, y_input224 + y_input224_err, color=input224_color, alpha=fill_in_alpha)

    # l_pad = plt.plot(x_grid, y_pad, color=pad_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Pad Prompt", alpha=pad_alpha)
    # plt.fill_between(x_grid, y_pad - y_pad_err, y_pad + y_pad_err, color=pad_color, alpha=fill_in_alpha)

    # l_fix = plt.plot(x_grid, y_fix, color=fix_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Fix Prompt", alpha=fix_alpha)
    # plt.fill_between(x_grid, y_fix - y_fix_err, y_fix + y_fix_err, color=fix_color, alpha=fill_in_alpha)

    # l_random = plt.plot(x_grid, y_random, color=random_color, marker='o', markevery=markevery, linestyle='-',
    #                   linewidth=linewidth,
    #                   markersize=markersize, label="Random Prompt", alpha=random_alpha)
    # plt.fill_between(x_grid, y_random - y_random_err, y_random + y_random_err, color=input224_color, alpha=fill_in_alpha)


    lbest = plt.axhline(y=y_best, color=best_color, linestyle='--', linewidth=3, alpha=best_alpha,
                        label="Best Winning Ticket")
    # lPFTT = plt.plot(x_grid, y_PFTT, color=PFTT_color, marker='*', markevery=markevery, linestyle='-',
    #                  linewidth=linewidth,
    #                  markersize=markersize + 4, label="PFTT", alpha=PFTT_alpha)
    # plt.fill_between(x_grid, y_PFTT - y_PFTT_err, y_PFTT + y_PFTT_err, color=PFTT_color, alpha=fill_in_alpha)

    plt.ylim([y_min, y_max])
    plt.xlim(0, 100)

    plt.legend(fontsize=fontsize - 15, loc=3, fancybox=True, shadow=True, framealpha=1.0, borderpad=0.3)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x_grid, x_sparsity_list, rotation=0, fontsize=fontsize)
    plt.xscale("linear")
    plt.yticks(fontsize=fontsize)

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    # plt.twinx()
    # y_time_label = "Time Consumption (min)"
    # linewidth = 2
    # time_linestyle = '-.'
    # tIMP = plt.plot(x_grid, y_IMP_time, color=IMP_color, alpha=IMP_alpha, label="IMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    # tBiP = plt.plot(x_grid, y_BiP_time, color=BiP_color, alpha=BiP_alpha, label="BiP", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # tPFTT = plt.plot(x_grid, y_PFTT_time, color=PFTT_color, alpha=PFTT_alpha, label="PFTT", linestyle=time_linestyle,
    #                  linewidth=linewidth)
    # tHydra = plt.plot(x_grid, y_hydra_time, color=hydra_color, alpha=hydra_alpha, label="Hydra Global",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # tGrasp = plt.plot(x_grid, y_Grasp_time, color=Grasp_color, alpha=Grasp_alpha, label="Grasp",
    #                   linestyle=time_linestyle, linewidth=linewidth)
    # tOMP = plt.plot(x_grid, y_OMP_time, color=OMP_color, alpha=OMP_alpha + 0., label="OMP", linestyle=time_linestyle,
    #                 linewidth=linewidth)
    #
    # plt.xlabel(x_label, fontsize=fontsize)
    # plt.ylabel(y_time_label, fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.ylim(0, (int(max(y_IMP_time) / 100) + 1) * 100)
    plt.savefig(f"pic/ablation_vp/{title}.pdf")
    plt.show()
    plt.close()










