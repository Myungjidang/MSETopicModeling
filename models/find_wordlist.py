import pandas as pd
import numpy as np
import tqdm
import pickle
import os

from utils.labels import label_centroid
from utils.nearest import docx_nearest, word_nearest
from utils.arguments import get_args
from utils.preprocess import make_base, docx_number
from utils.kalman import kalmanfilter
from utils.VAR import var
from utils.average import average
from utils.visualization import visual_scala, visual_random_scala, visual_relative_scala
from utils.visualization import visual_vector


def datasave(data, save_path, kvm, sv, args):
    if kvm == "measurements":
        with open(f"{save_path}/{kvm}/{sv}/{args.start_date}_{args.end_date}", 'wb') as f:
            pickle.dump(data, f)
    elif kvm == "var" or kvm == "kalman":
        with open(f"{save_path}/{kvm}/{sv}/{args.start_date}_{args.end_date}_iter500", 'wb') as f:
            pickle.dump(data, f)

def make_docx_v(args):
    country_list = ["cn", "de", "en", "jp", "kr"]
    docx_list = []
    for country in country_list:
        docx_v = np.load(f"{args.datapath}/multi_9_{country}_doc_v.npy")
        docx_list.extend(list(docx_v))
    # print(np.array(docx_list).shape)
    return np.array(docx_list)


def first_excute(save_path):
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        return
    sv = ["scala", "vector"]
    kv = ["kalman", "var"]
    folder_names = ["measurements", "var", "kalman", "average", "images"]

    for names in folder_names:
        for i in range(2):
            dir_name = f"{save_path}/{names}/{sv[i]}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    
    # os.makedirs(f"{save_path}/images/scala_amount")
    # os.makedirs(f"{save_path}/images/scala_onescore")
    # os.makedirs(f"{save_path}/images/vector/kalman")
    # os.makedirs(f"{save_path}/images/vector/var")
    pass

def main():
    datapath = "../patent_data/multilingual_v2"
    save_path = "../result_stack_768"

    n_cluster = 30
    args = get_args()
    custompath = f"./{n_cluster}"
    kalman_mode = "origin"

    docx_v = make_docx_v(args)
    docx_v_df = pd.DataFrame(docx_v)

    # first_excute(save_path)

    # label, centroids = label_centroid(docx_v_df, "make") # first excute
    label, centroids = label_centroid(docx_v_df, "use")
    label_df = pd.DataFrame(label, columns=["label"])

    (
        scala_motion_controls,
        scala_measurements,
        _,
        vector_measurements,
    ) = make_base(custompath, args, label, centroids)
    # docx_num_list = docx_number(custompath, args, label, centroids)
    # print(scala_measurements) 
    # scala_measurements: (n_cluster, timeseries length)
    # vector_measurements: (n_cluster, timeseries length, vector dim)
    # datasave(scala_measurements, save_path, "measurements", "scala", args)
    # datasave(vector_measurements, save_path, "measurements", "vector", args)

    # Nearest (O)
    # docx_nearest(datapath, n_cluster, label_df, docx_v_df, centroids, "origin")
    # word_nearest(datapath, centroids, "origin")

    # Kalman Filter (Scala(O), Vector(O))
    # kalman_s = kalman_filter_scala(scala_motion_controls, scala_measurements, args)
    # kalman_v = kalmanfilter(scala_measurements, vector_measurements, args, kalman_mode)
    # # # word_nearest(datapath, kalman_v, "kalman")
    # datasave(kalman_s, save_path, "kalman", "scala", args)
    # datasave(kalman_v, save_path, "kalman", "vector", args)

    # # VAR (O)
    var_s, var_v = var(scala_measurements, vector_measurements, args)
    # datasave(var_s, save_path, "var", "scala", args)
    # datasave(var_v, save_path, "var", "vector", args)
    word_nearest(datapath, var_v, "var")
    
    # # # Average (O)
    # for i in [1,3,5]:
    #     average(scala_measurements, i, "scala", args)
    #     average(vector_measurements, i, "vector", args)
    # #     pass

    # Visualization 
    # visual_scala(scala_measurements, args)
    # visual_random_scala(scala_measurements, args)
    # visual_relative_scala(scala_measurements, docx_num_list, args)
    # visual_vector(vector_measurements, centroids, args, "var")
    


if __name__ == "__main__":
    main()
# 