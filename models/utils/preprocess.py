import operator
import pandas as pd
import numpy as np
import pickle
import os.path
import os

def datasave(data, save_path, kvm, sv, args):
    if kvm == "measurements":
        with open(f"{save_path}/{kvm}/{sv}/{args.start_date}_{args.end_date}", 'wb') as f:
            pickle.dump(data, f)
    elif kvm == "var":
        with open(f"{save_path}/{kvm}/{sv}/{args.start_date}_{args.end_date}", 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(f"{save_path}/{kvm}/{sv}/{args.start_date}_{args.end_date}", 'wb') as f:
            pickle.dump(data, f)


def make_docx_v(args, country_list):
    # country_list = ["cn", "de", "en", "jp", "kr"]
    country_list = ["chinese", "english", "german", "japanese", "korean"]
    docx_list = []
    for country in country_list:
        docx_v = np.load(f"{args.datapath}save_data/prj_doc_{country}.npy")
        docx_list.extend(list(docx_v))
    return np.array(docx_list)


def first_excute(save_path):
    sv = ["scala", "vector"]
    folder_names = ["measurements", "var", "average", "images"]
    image_scala_folder_names = ["total", "amount","onescore"]

    for names in folder_names:
        for i in range(2):
            dir_name = f"{save_path}/{names}/{sv[i]}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    for names in image_scala_folder_names:
        dir_name = f"{save_path}/images/scala/{names}"
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    if not os.path.exists(f"{save_path}/images/vector/var"):
        os.makedirs(f"{save_path}/images/vector/var")
    if not os.path.exists(f"{save_path}/wordlist"):
        os.makedirs(f"{save_path}/wordlist")
    if not os.path.exists(f"{save_path}/docxlist"):
        os.makedirs(f"{save_path}/docxlist")


def csv_to_numpy(csvpath, csv_type):
    if csv_type is "centroid":
        df = pd.read_csv(csvpath)
        header = df.columns.to_numpy()
        header = np.reshape(header, ((1,) + header.shape))
        new_df = df.to_numpy()
        new_df = np.concatenate((header, new_df), axis=0)
    elif csv_type is "kalman":
        df = csvpath
        df = df.drop("time", axis=1)
        df = df.drop("cluster", axis=1)
        new_df = df.to_numpy()
    elif csv_type is "word":
        df = pd.read_csv(csvpath)
        df = df.drop("word_list", axis=1)
        new_df = df.to_numpy()
    else:
        df = pd.read_csv(csvpath, index_col=[0])
        df = df.drop("time", axis=1)
        new_df = df.to_numpy()
    return new_df


def find_docx_centroid(docx):
    # 여러 docx의 중심 vector를 찾는 함수
    # print(docx.mean())
    docx_np = csv_to_numpy(docx, "kalman")
    centroid = docx_np.mean(axis=0)
    return centroid

def docx_number(args, label, centroids):
    datapath = args.datapath
    start = args.start_date
    end = args.end_date

    origin_df = make_cluster_information_csv(
        datapath, label, centroids
    )
    docx_number_list = []
    for time in range(start, end + 1):
        docx_number_list.append(len(origin_df[origin_df["time"]==time]))
    return docx_number_list

def df_time(x):
    x = int(x[:4])
    return x

def make_origin_df(datapath):
    country_list = ["chinese", "english", "german", "japanese", "korean"]
    docx_list = []
    time_list = []
    for country in country_list:
        docx_v = np.load(f"{datapath}save_data/prj_doc_{country}.npy")
        docx_list.extend(list(docx_v))
    origin_df = pd.DataFrame(np.array(docx_list))

    for country in country_list:
        c_time = pd.read_csv(f"{datapath}/{country}_data.csv")["출원일"]
        c_time = list(c_time.apply(df_time))
        time_list.extend(c_time)
    time_df = pd.DataFrame(np.array(time_list), columns=["time"])
    origin_df = pd.concat([time_df, origin_df], axis=1)

    return origin_df


def make_base_scala(docx_count_list):
    """
    cluster의 문서 개수를 기반으로, 차년도 문서가 몇 개 생성될지 예측하는 task
    해당 task에 사용하기 위해 변수 반환
    input : docx_count_list(n_cluster, len_time)
    output : measurement, motion_control
            measurement(list) = cluster 내부의 연도별 문서 개수
                measurement[0](list) = 0번째 cluster의 연도별 문서 개수
            motion_control(list) = 연도별 문서 개수의 변화량(scala)
    """
    motion_controls = [[] for i in range(len(docx_count_list))]
    measurements = docx_count_list

    for idx1, cluster in enumerate(measurements):
        for idx2 in range(len(cluster) - 1):
            motion_controls[idx1].append(cluster[idx2 + 1] - cluster[idx2])

    motion_controls = np.array(motion_controls)
    measurements = np.array(measurements)

    return motion_controls, measurements


def make_base_vector(centroid_list):
    """
    cluster의 중심 좌표를 기반으로, 차년도 중심 좌표가 어디에 생성될지 예측하는 task
    해당 task에 사용하기 위해 변수 반환
    input :     centroid_list(n_cluster, len_time, vector dim)
    output :    measurement, motion_control
                motion_control(list) =  전체 데이터의 중심좌표와의 연도별 중심좌표 사이의 변화량(vector)
                                        (n_cluster, len_time, vector dim)
                measurement(list) = cluster 내부의 연도별 중심 좌표 (n_cluster, len_time, vector dim)
    """
    motion_controls = [[] for i in range(len(centroid_list))]
    measurements = centroid_list

    for idx1, cluster in enumerate(measurements):
        for idx2 in range(len(cluster) - 1):
            motion_controls[idx1].append(cluster[idx2 + 1] - cluster[idx2])

    motion_controls = np.array(motion_controls)
    return motion_controls, measurements


def cluster_to_time_centroid(args, origin_df):
    """
    n번째 cluster에 지정한 time dataset이 얼마나 존재하는지 셈한다.
    input :     n_cluster
    output :    centroid_list : [n번째 cluster][m time의 문서 중심]
                docx_count_list : [n번째 cluster][m time의 문서 개수]
    """
    centroid_list = [[] for i in range(args.n_cluster)]
    docx_count_list = [[] for i in range(args.n_cluster)]

    for i in range(args.n_cluster):  # n번째 cluster에 대해
        n_cluster = origin_df[origin_df["cluster"] == i]
        for j in range(args.start_date, args.end_date + 1):  # m번째 time의 문서 개수 / 중심(mean)
            time_docx = n_cluster[n_cluster["time"] == j]
            docx_count_list[i].append(len(time_docx))
            centroid_list[i].append(find_docx_centroid(time_docx))
    centroid_list = np.array(centroid_list)
    docx_count_list = np.array(docx_count_list)
    return centroid_list, docx_count_list

def make_cluster_information_csv(datapath, label, centroids):
    """
    origin dataset에 KMeans를 거친 label을 추가한다.
    input : datapath, 
            label, 
            centroids
    output : origin_df
    """
    origin_df = make_origin_df(datapath)
    label_df = pd.DataFrame(label, columns=["label"])
    origin_df.insert(1, "cluster", label_df)

    return origin_df

def make_base(args, label, centroids):
    """
    통합된 데이터로부터 시간 정보를 사용해 분류하는 preprocessing
    input :     args : main.py 시작시 초기화된 매개변수를 사용하기 위한 변수
                label : 
                centroids :
    output :    scala_measurements : 각 Topic의 시간대별 문서량
                vector_measurements : 각 Topic의 시간대별 문서 중심점
    """
    origin_df = make_cluster_information_csv(
        args.datapath, label, centroids
    )
    centroid_list, docx_count_list = cluster_to_time_centroid(
        args, origin_df
    )
    # centroid_list[n][m] : n번 cluster의 m time에서의 중심점
    # docx_count_list[n][m] : n번 cluster의 m time의 문서 개수
    scala_motion_controls, scala_measurements = make_base_scala(docx_count_list)
    vector_motion_controls, vector_measurements = make_base_vector(centroid_list)

    return (
        scala_measurements,
        scala_motion_controls,
        vector_measurements,
        vector_motion_controls
    )
