import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from IPython.display import HTML
from itertools import cycle

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


def get_approaches_results(datasets, column_mean, column_std, group_by_th=False, items_num=1000):
    mean = []
    std = []
    # initial
    if group_by_th == True:
        ths = datasets[0]["threshold"]
        for th in ths:
            for data in datasets:
                data = data[data["threshold"] == th]
                if (column_mean == 'cost'):
                    vals = data[column_mean] / data['cost_ratio'][0] / items_num
                    for v in vals:
                        mean.append(v)
                    vals_std = data[column_std] / data['cost_ratio'][0] / items_num
                    for s in vals_std:
                        std.append(s)
                else:
                    vals = data[column_mean]
                    for v in vals:
                        mean.append(v)
                    vals_std = data[column_std]
                    for s in vals_std:
                        std.append(s)
    else:
        for data in datasets:
            if (column_mean == 'cost'):
                vals = data[column_mean] / data['cost_ratio'] / items_num
                for v in vals:
                    mean.append(v)
                vals_std = data[column_std] / data['cost_ratio'] / items_num
                for s in vals_std:
                    std.append(s)
            else:
                vals = data[column_mean]
                for v in vals:
                    mean.append(v)
                vals_std = data[column_std]
                for s in vals_std:
                    std.append(s)

    return mean, std


def get_group_colors(groups, subgroups):
    # colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
    return np.concatenate([[np.concatenate(np.random.rand(3, 1))] * subgroups for x in range(groups)])


def concatenate(list1, list2):
    result = []
    for elem1 in list1:
        result.append(elem1)

    for elem2 in list2:
        result.append(elem2)

    return result


def get_total_results(list1, list2, column, mv_num=None, cost_ratio=None, decision_fn=None, threshold=None,
                      class_fn=None, c=None, e=None):
    if mv_num != None:
        return concatenate(list1[list1['votes'] == mv_num][column].values,
                           list2[list2['votes'] == mv_num][column].values)
    elif c != None:
        return concatenate(list1[list1['c'] == c][list1['e'] == e][column].values,
                           list2[list2['c'] == c][list2['e'] == e][column].values)
    elif class_fn != None:
        return concatenate(list1[list1['class_fn'] == class_fn][list1['decision_fn'] == decision_fn][
                               list1['cost_ratio'] == cost_ratio][list1['threshold'] == threshold][column].values,
                           list2[list2['class_fn'] == class_fn][list2['decision_fn'] == decision_fn][
                               list2['cost_ratio'] == cost_ratio][list2['threshold'] == threshold][column].values)
    elif decision_fn != None:
        return concatenate(list1[list1['decision_fn'] == decision_fn][list1['cost_ratio'] == cost_ratio][
                               list1['threshold'] == threshold][column].values,
                           list2[list2['decision_fn'] == decision_fn][list2['cost_ratio'] == cost_ratio][
                               list2['threshold'] == threshold][column].values)
    else:
        return concatenate(list1[column].values,
                           list2[column].values)


def plot_elems_bars(elements, columns):
    # Figure 1
    ind = np.arange(len(elems))
    width = .1
    xticks_ind = ind + width / 2
    xticks_rotation = 70
    xticks_names = elems['name']

    plt.figure(num=1, figsize=(40, 30), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(bottom=.001)

    for key, column in enumerate(columns):
        # plot_num = 320 + key
        plt.subplot(3, 2, key + 1)
        column_mean = elements[column]
        column_std = elements[f"{column}_std"]

        p1 = plt.bar(ind, column_mean, width, yerr=column_std, zorder=3, color='red', edgecolor='black')
        plt.ylabel(column)
        plt.title(column, fontweight="bold", fontsize=40)
        plt.xticks(xticks_ind, xticks_names, rotation=xticks_rotation, fontsize=30)
        plt.grid(zorder=0)
        if max(elements[column]) > 1:
            plt.yticks(np.arange(0, max(elements[column]) + 3, 1), fontsize=20)
        else:
            plt.yticks(np.arange(0, 1.1, .1), fontsize=20)
        # plt.savefig("/Users/pmaglione/Repos/adaptive-pomdp-solutions/dai_pomdp/charts/test", bbox_inches = 'tight', pad_inches = 0)

        plt.plot(xticks_ind, column_mean, "k--")

    # end for
    plt.tight_layout()
    plt.show()


lines = ["-", "--", "-.", ":"]


def plot_elems_lines(elems, x_values, columns, ylabel, legends):
    xticks_ind = np.arange(len(x_values))
    plt.figure(num=1, figsize=(20, 20), facecolor='w', edgecolor='k')
    plt.grid(zorder=0)

    for key, column in enumerate(columns):
        linecycler = cycle(lines)
        plt.subplot(3, 2, key + 1)
        column_std = f'{column}_std'

        for elem in elems:
            plt.errorbar(xticks_ind, column, column_std, data=elem, linestyle=next(linecycler), marker='o',
                         markersize=5)
        # end for

        plt.xticks(xticks_ind, x_values)
        plt.xlabel(ylabel)
        plt.ylabel(column.capitalize())
        plt.grid()
        plt.title(column.capitalize(), fontweight="bold", fontsize=20)
        plt.legend(legends)

    plt.tight_layout()
    plt.show()


def plot_elems_comparisson(elements1, elements2, elem1_name, elem2_name, columns):
    width = 0.1  # the width of the bars: can also be len(x) sequence
    names1 = [f"{elem1_name}-cost={elem[1]['wrong_cost']}-after={elem[1]['estimate_after']}" for elem in
              elements1.iterrows()]
    names2 = [f"{elem2_name}-cost={elem[1]['wrong_cost']}-after={elem[1]['estimate_after']}" for elem in
              elements2.iterrows()]

    xticks_names = []
    wc_1 = [elem[1]['wrong_cost'] for elem in elements1.iterrows()]
    ea_1 = [elem[1]['estimate_after'] for elem in elements1.iterrows()]
    wc_2 = [elem[1]['wrong_cost'] for elem in elements2.iterrows()]
    ea_2 = [elem[1]['estimate_after'] for elem in elements2.iterrows()]

    for elem_ind in range(len(elements1[columns[0]])):
        xticks_names.append(f"base-cost={wc_1[elem_ind]}-after={ea_1[elem_ind]}")
        xticks_names.append(f"uncl-cost={wc_2[elem_ind]}-after={ea_2[elem_ind]}")

    xticks_rotation = 90
    plt.figure(num=1, figsize=(40, 30), dpi=80, facecolor='w', edgecolor='k')
    plt.grid(zorder=0)

    for key, column in enumerate(columns):
        plt.subplot(3, 2, key + 1)
        N = len(elements1[columns[0]])
        ind_1 = np.arange(0, N, 1)
        ind_2 = np.arange(0.5, N, 1)

        ind_total = np.arange(0, N, .5)

        column_mean_1 = elements1[column]
        column_std_1 = elements1[f"{column}_std"]

        column_mean_2 = elements2[column]
        column_std_2 = elements2[f"{column}_std"]

        p1 = plt.bar(ind_1, column_mean_1, width, yerr=column_std_1)
        p2 = plt.bar(ind_2, column_mean_2, width, yerr=column_std_2)

        plt.ylabel(column)
        plt.title(f'Comparison {column} | {elem1_name} - {elem2_name}', fontsize=30)
        plt.xticks(ind_total, xticks_names, rotation=xticks_rotation, fontsize=18)

        if max(elements1[column]) >= max(elements2[column]):
            elements_max = elements1
        else:
            elements_max = elements2

        if max(elements_max[column]) > 1:
            plt.yticks(np.arange(0, max(elements_max[column]) + 3, 1), fontsize=30)
        else:
            plt.yticks(np.arange(0, 1.1, .1), fontsize=15)

        plt.legend((p1[0], p2[0]), (elem1_name, elem2_name), fontsize=30)

        plt.grid(zorder=0)

        # end for
    plt.tight_layout()
    plt.show()

def get_beta(fnc, fpc):
    fnc = abs(fnc)
    fpc = abs(fpc)
    if fnc == fpc:
        return 1
    elif fnc > fpc:
        return fnc
    else:
        return fnc / fpc