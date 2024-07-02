import pandas as pd
import numpy as np
from tqdm import tqdm


def get_neighbors(centroid, location, words, num_neighbors):
    distances = []
    for idx, word in enumerate(words):
        dist = np.linalg.norm(centroid - word)
        distances.append((location[idx], dist))
    distances.sort(key=lambda tup: tup[1])
    distances = distances[:num_neighbors]

    neighbor = [i[0] for i in distances]
    distance = [i[1] for i in distances]
    return neighbor, distance


def get_neighbors_docx(centroid, location, words, num_neighbors):
    distances = []
    # print(words.iloc[0].tolist())
    for idx in range(len(words)):
        word = words.iloc[idx].tolist()
        dist = np.linalg.norm(centroid - word)
        distances.append((location[idx], dist))
    distances.sort(key=lambda tup: tup[1])
    distances = distances[:num_neighbors]

    neighbor = [i[0] for i in distances]
    distance = [i[1] for i in distances]

    return neighbor, distance


def docx_nearest(args, centroids, label_df, docx_v_df, country_list, num_nearest, mode):
    total_docs_list = []

    for country in country_list:
        docx_df = pd.read_csv(f"{args.datapath}/{country}_data.csv")
        docx_df["요약+대표청구항"] = docx_df["요약"] + docx_df["대표청구항"]
        docx_text_df = docx_df[["요약+대표청구항"]]
        total_docs_list.extend(docx_text_df.values.tolist())
    total_docx_df = pd.DataFrame(total_docs_list, columns=["요약+대표청구항"])

    nearest_docs_list = []
    for i, centroid in enumerate(centroids):
        location_cluster = label_df.index[label_df["label"] == i].tolist()
        docx_cluster = docx_v_df.iloc[location_cluster]

        neighbors, _ = get_neighbors_docx(
            centroid, location_cluster, docx_cluster, num_nearest
        )
        nearest_docs_list.append(
            total_docx_df.iloc[neighbors]["요약+대표청구항"].values.tolist()
        )
    total_docs_df = pd.DataFrame(nearest_docs_list)
    total_docs_df.to_csv(f"{args.save_path}/docxlist/{mode}.csv", index=False)

def docx_nearest_onecountry(args, centroids, label_df, country_list, num_nearest, mode):
    # TODO : label 오류로 인한 index out of bound
    return
    for country in country_list:
        docx_df = pd.read_csv(f"{args.datapath}/multi_{country}_dataset.csv")
        docx_v_df = pd.DataFrame(np.load(f"{args.datapath}/multi_9_{country}_doc_v.npy"))
        docx_df["요약+대표청구항"] = docx_df["요약"] + docx_df["대표청구항"]
        docx_text_df = docx_df[["요약+대표청구항"]]

        nearest_docs_list = []
        for i, centroid in enumerate(centroids):
            location_cluster = label_df.index[label_df["label"] == i].tolist()
            docx_cluster = docx_v_df.iloc[location_cluster]

            neighbors, _ = get_neighbors_docx(
                centroid, location_cluster, docx_cluster, num_nearest
            )
            nearest_docs_list.append(
                docx_text_df.iloc[neighbors]["요약+대표청구항"].values.tolist()
            )
        total_docs_df = pd.DataFrame(nearest_docs_list)
        total_docs_df.to_csv(f"{args.save_path}/docxlist/{country}_{mode}.csv", index=False)


def word_nearest(args, centroids, n_word, country_list, mode):
    word_v_list = []
    word_list_list = []
    for country in country_list:
        word_v = np.load(f"{args.datapath}/save_data/prj_word_{country}.npy")
        wordlist_v = np.load(f"{args.datapath}/save_data/top_word_list_{country}.npy")
        word_v_list.extend(list(word_v))
        word_list_list.extend(list(wordlist_v))

    word_v_df = pd.DataFrame(word_v_list)
    word_list_df = pd.DataFrame(word_list_list, columns=["word"])

    total_word_list = []
    locations = [i for i in range(len(word_v_df))]
    for idx, centroid in enumerate(tqdm(centroids)):
        neighbors, _ = get_neighbors(centroid, locations, word_v_list, n_word)
        total_word_list.append(word_list_df.iloc[neighbors]["word"].values.tolist())
    total_word_list = pd.DataFrame(total_word_list)
    total_word_list.to_csv(f"{args.save_path}/wordlist/{mode}.csv", index=False)

def word_nearest_onecountry(args, centroids, n_word, country_list, mode):
    for country in country_list:
        word_v = np.load(f"{args.datapath}save_data/prj_word_{country}.npy")
        wordlist_v = np.load(f"{args.datapath}save_data/top_word_list_{country}.npy")
        word_v_df = pd.DataFrame(word_v)
        word_list_df = pd.DataFrame(wordlist_v, columns=["word"])
        
        total_word_list = []
        locations = [i for i in range(len(word_v_df))]
        for idx, centroid in enumerate(tqdm(centroids)):
            neighbors, _ = get_neighbors(centroid, locations, word_v, n_word)
            total_word_list.append(word_list_df.iloc[neighbors]["word"].values.tolist())
        total_word_list = pd.DataFrame(total_word_list)
        total_word_list.to_csv(f"{args.save_path}/wordlist/{country}_{mode}.csv", index=False)