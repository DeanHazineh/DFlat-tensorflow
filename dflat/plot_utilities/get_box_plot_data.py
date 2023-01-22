import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        # dict1["label"] = labels[i]
        dict1["whislo"] = bp["whiskers"][i * 2].get_ydata()[1]
        dict1["q1"] = bp["boxes"][i].get_ydata()[1]
        dict1["med"] = bp["medians"][i].get_ydata()[1]
        dict1["q3"] = bp["boxes"][i].get_ydata()[2]
        dict1["whishi"] = bp["whiskers"][(i * 2) + 1].get_ydata()[1]
        rows_list.append(dict1)

    return rows_list  # , pd.DataFrame(rows_list)
