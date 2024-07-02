import os
import pickle
import sklearn.cluster as cluster

def clustering(origin_df, n_cluster):
    """
    K-Means를 사용한 Clustering 함수
    input :     origin_df
                n_cluster
    output :    label
                centroid
    """
    print("A task that takes time to execute. Hold on a minute, please.")
    model = cluster.KMeans(n_cluster, n_init=100, max_iter=1000)
    label = model.fit_predict(origin_df)
    centroids = model.cluster_centers_
    return label, centroids


def label_centroid(origin_df, n_cluster):
    """
    K-Means를 사용해 label과 centroid 정보를 도출하고 저장하는 함수
    input :     original_df
                n_cluster
    output :    label
                centroid
    """
    path = f"./tmp_data/label_centroid_information/{n_cluster}"
    if not os.path.exists(path):
            os.makedirs(path)
            label, centroids = clustering(origin_df, n_cluster)
            with open(f"{path}/label", "wb") as fp:
                pickle.dump(label, fp)
            with open(f"{path}/centroids", "wb") as fp:
                pickle.dump(centroids, fp)
    else:
        with open(f"{path}/label", "rb") as fp:
            label = pickle.load(fp)
        with open(f"{path}/centroids", "rb") as fp2:
            centroids = pickle.load(fp2)
    return label, centroids