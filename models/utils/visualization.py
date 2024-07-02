import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import pandas as pd
import numpy as np
import pickle

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

def make_tSNE(data, n_components):
    model = TSNE(n_components=n_components, init="pca")
    shrinked_data = model.fit_transform(data)
    return shrinked_data

def s_data(args, mode, avg_num=0):
    df = pd.DataFrame()
    if "average" in mode:
        for date in range(2017, args.end_date):
            path = f"{args.save_path}/{mode}/scala/{date}_{avg_num}.csv"
            data = pd.read_csv(path)
            df[f"{date}"] = data["predict"]
    else:
        for date in range(2017, args.end_date):
            path = f"{args.save_path}/{mode}/scala/{args.start_date}_{date}"
            with open(file=path, mode="rb") as f:
                data = pickle.load(f)
            
            df[f"{date}"] = data["predict"]
        
    return df.values.tolist()

def visual_scala_relative(args, measurements, docx_num_list):
    """
    연도별 문서 누계
    """
    # palette = sns.color_palette("husl", 30)
    palette = ["red", "blue"]
    range_list = []
    total_docx_scala = []
    plt.cla()
    for i in range(args.n_cluster):
        timeseries = measurements[i]
        range_list.append((timeseries[0] / docx_num_list[0]) - (timeseries[-1] / docx_num_list[-1]))

    range_min = range_list.index(min(range_list))
    range_max = range_list.index(max(range_list))
    imagelist = [range_min, range_max]

    # for idx in range(args.n_cluster):
    for i, idx in enumerate(imagelist):
        plt.cla()
        timeseries = measurements[idx] / docx_num_list
        x = np.arange(len(timeseries)) * 2
        years = [i for i in range(args.start_date, args.end_date + 1)]
        plt.xticks(x, years)
        plt.plot(x, timeseries, color = "#c68a12", label=f"Topic {idx}", marker='o')
        plt.savefig(f"{args.save_path}/images/scala/amount/{args.end_date}_{idx}.png")
    plt.legend(loc=2, fontsize="5")
    # plt.savefig(f"{args.save_path}/images/scala_amount/{args.end_date}_total.png")



def visual_scala_onescore(args, measurements):
    palette = sns.color_palette("husl", 30)
    var_predict_data = s_data(args, "var")
    # kalman_predict_data = s_data(args, "kalman")
    values = [i for i in range(args.n_cluster)]
    # plt.grid(True, color="#f5f5f5")

    for idx, rv in enumerate(values):
        plt.cla()
        timeseries = measurements[rv]
        var_data = var_predict_data[rv]
        # kalman_data= kalman_predict_data[rv]
        years = [i for i in range(args.start_date, args.end_date + 1)]
        for j in range(len(years)):
            years[j] = "" if j % 3 != 0 else years[j]
        x = np.arange(len(years)) * 2
        p_x = x[-len(var_data):]
        plt.xlabel(f"Forecast of next year's document volume")
        plt.plot(x, timeseries, color = palette[idx], label=f"Topic {rv}", marker='o')
        plt.scatter(p_x, var_data, color = palette[idx], s=8)
        # plt.scatter(p_x, kalman_data, color = palette[idx], s=8)
        plt.xticks(x, years)
        plt.tick_params(
            axis="x",
            length=2,
            pad=6,
            labelsize=5
        )
        plt.legend(loc=2, fontsize="5")
        plt.savefig(f"{args.save_path}/images/scala/onescore/{rv}.png")

def visual_scala_total(args, measurements):
    palette = sns.color_palette("bright", 15)
    var_predict_data = s_data(args, "var")
    average1 = s_data(args, "average", 1)
    average3 = s_data(args, "average", 3)
    average5 = s_data(args, "average", 5)
    for i in range(args.n_cluster):
        plt.cla()
        # plt.grid(True, color="#f5f5f5")
        timeseries = measurements[i]
        var_data = var_predict_data[i]

        years = [i for i in range(args.start_date, args.end_date + 1)]
        for j in range(len(years)):
            years[j] = "" if j % 3 != 0 else years[j]
        
        x = np.arange(len(years)) * 2
        p_x = x[-len(var_data):]
        plt.xlabel(f"Forecast of next year's document volume | cluster {i}")
        plt.plot(x, timeseries, color = palette[0], label="actual data", marker='o')
        plt.scatter(p_x, var_data, color = palette[2], s=30, label="predict(var)", marker="s")
        plt.scatter(p_x, average1[i], color = palette[6], s=5, label="predict(average1)")
        plt.scatter(p_x, average3[i], color = palette[8], s=5, label="predict(average3)")
        plt.scatter(p_x, average5[i], color = palette[10], s=5, label="predict(average5)")
        plt.xticks(x, years)
        plt.tick_params(
            axis="x",
            length=2,
            pad=6,
            labelsize=5
        )
        plt.legend(loc=2)
        plt.savefig(f"{args.save_path}/images/scala/total/{args.end_date}_{i}.png")
        

def v_data(args, kv):
    v_datalist = []
    data_list = []
    predict_start = 2017
    for date in range(predict_start, args.end_date+1):
        path = f"{args.save_path}/{kv}/vector/{args.start_date}_{date}"
        with open(file=path, mode="rb") as f:
            data = pickle.load(f)
            data_list.append(list(data))
    
    for tensors in zip(*data_list):
        v_datalist.extend(tensors)
    return np.array(v_datalist)

def visual_vector(args, measurements, centroids, kv):
    time_color = sns.color_palette("pastel", args.n_cluster).as_hex()
    predict_color = sns.color_palette("bright", args.n_cluster).as_hex()
    bright_color = sns.color_palette("bright", args.n_cluster).as_hex()
    time_length = args.end_date - args.start_date + 1
    new = np.reshape(measurements, (args.n_cluster*time_length, 768))

    
    if kv == "var":
        predict = v_data(args, "var")
    else:
        predict = v_data(args, "kalman")
    
    total_data = np.concatenate((new, predict), axis=0)
    shrinked_total_data = make_tSNE(total_data, 2)
    
    
    time_data = shrinked_total_data[:args.n_cluster*time_length]
    predict = shrinked_total_data[args.n_cluster*time_length:]

    
    len_op = int(predict.shape[0] / args.n_cluster)
    
    for idx in range(args.n_cluster):
        plt.cla()
        p_start = len_op * idx
        t_start = idx * time_length
        cluster_predict = pd.DataFrame(predict[p_start:p_start+len_op], columns=["x", "y"])
        cluster_time = pd.DataFrame(time_data[t_start:t_start+time_length],columns=["x", "y"])       
        # predict
        plt.plot(
            cluster_predict["x"],
            cluster_predict["y"],
            color=bright_color[4],
            alpha=0.6,
        )
        plt.scatter(
            cluster_predict["x"],
            cluster_predict["y"],
            color=bright_color[4],
            s=8
        )
        # actual timeline
        plt.plot(
            cluster_time["x"],
            cluster_time["y"],
            color=bright_color[3],
            alpha=0.6,
        )
        plt.scatter(
            cluster_time["x"],
            cluster_time["y"],
            color=bright_color[3],
            s=8
        )
        cluster_length = len(cluster_predict["x"])
        time_length = len(cluster_time["x"])

        plt.scatter(
            cluster_predict["x"][0],
            cluster_predict["y"][0],
            color=bright_color[10],
            s=10
        ) # 시작점
        plt.scatter(
            cluster_time["x"][time_length - cluster_length],
            cluster_time["y"][time_length - cluster_length],
            color=bright_color[10],
            s=10
        )
        # annotate
        plt.annotate("2017", (cluster_predict["x"][0], cluster_predict["y"][0]))
        plt.annotate("2017", (cluster_time["x"][time_length - cluster_length], cluster_time["y"][time_length - cluster_length]))

        plt.annotate(f"{args.end_date}", (cluster_predict["x"][cluster_length-1], cluster_predict["y"][cluster_length-1]))
        plt.annotate(f"{args.end_date}", (cluster_time["x"][time_length-1], cluster_time["y"][time_length-1]))

        plt.savefig(f"{args.save_path}/images/vector/{kv}/{idx}.png")



    