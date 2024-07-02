import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from gensim import corpora
import gensim
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import aspose.words as aw
from collections import Counter
import re
import plotly.express as px


def make_fields(country_data):
    code_list = ['H01L', 'G09G', 'G02F', 'G06F', 'C09K', 'G02B', 'C07D', 'G09F', 'H04N', 'B32B',
                 'H05K', 'H04R', 'G03F', 'C23C', 'C07C', 'H05B', 'C09J', 'G06K', 'G01N', 'H01B',
                 'C08L', 'C07F', 'G06T', 'G01R', 'B23K', 'H04M', 'C09D', 'C08K', 'B41J', 'C03B',
                 'F21Y', 'C03C', 'H01S', 'H04L', 'C08G', 'C08J', 'B29C', 'C09B', 'F21K', 'C08F',
                 'G03H', 'H01R', 'H03K', 'B65G', 'C23F', 'H03M', 'F21V', 'H01J', 'B60K', 'G01T']
    code_list.reverse()
    country_fields = []
    for i in range(len(country_data)):
        country_fields.append(country_data['Original IPC All'][i])
    new_fields = []
    for string in country_fields:
        if type(string) != float:
            new_str = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", string)
            new_list = new_str.split(' ')
            new_fields.append(new_list)
        else:
            new_fields.append('')

    fields = []
    for new_field in new_fields:
        field = [x for i in code_list for x in new_field if i in x]
        if len(field) != 0:
            fields.append(field[0])
        else:
            fields.append('None')
    return fields



def make_document(country_data):
    country_documents = []
    for i in range(len(country_data)):
        country_documents.append(country_data['Documents'][i])
    return country_documents



def k_means_clustering(n_clusters, vector_list):
    print('------clustering 진행중------')
    k_means_prj = KMeans(n_clusters=n_clusters, n_init=50)
    k_means_prj.fit(vector_list)
    prj_topic_vector_list = k_means_prj.cluster_centers_
    return prj_topic_vector_list



def Document_number(n_clusters, country, prj_topic_vector_list, prj_country_doc_vector_list):
    prj_doc_ids_country_dist = {}
    country_doc_num = []
    for k in range(len(prj_country_doc_vector_list)):
        doc_vector_list_sort = []
        doc_vector_list_dist = []

        for i in range(n_clusters):
            doc_vector_list_dist.append(
                np.linalg.norm(prj_country_doc_vector_list[k] - np.array(prj_topic_vector_list[i])))
            doc_vector_list_sort.append(
                np.linalg.norm(prj_country_doc_vector_list[k] - np.array(prj_topic_vector_list[i])))
        doc_vector_list_sort.sort(reverse=False)

        prj_doc_ids_country_dist[f'{country}_Doc{k}'] = set([doc_vector_list_dist.index(doc_vector_list_sort[0])])

        country_doc_num.append(doc_vector_list_dist.index(doc_vector_list_sort[0]))
    return country_doc_num



def Topic_words(word_v, word_list, prj_topic_vector_list, n_clusters):
    T_topic_words_dict = {}
    T_topic_words_list = []
    T_topic_words_eucl_dict = {}
    for i in range(n_clusters):
        keyword_vector_list_sort = []
        keyword_vector_list_dist = []
        for k in range(len(word_v)):
            keyword_vector_list_dist.append(np.linalg.norm(prj_topic_vector_list[i] - np.array(word_v[k])))
            keyword_vector_list_sort.append(np.linalg.norm(prj_topic_vector_list[i] - np.array(word_v[k])))
        keyword_vector_list_sort.sort(reverse=False)
        keyword_list = []
        for j in range(n_clusters):
            keyword_number = keyword_vector_list_dist.index(keyword_vector_list_sort[j])
            keyword_list.append(word_list[keyword_number])

        T_topic_words_list.append(keyword_list)
        T_topic_words_dict[f'topic {i}'] = keyword_list[:5]
        T_topic_words_eucl_dict[f'topic {i}'] = keyword_vector_list_sort[:5]
    return T_topic_words_list, T_topic_words_eucl_dict, T_topic_words_dict



def Topic_docs(doc_v, documents, prj_topic_vector_list, fields, n_clusters):
    new_documents = []
    for i in range(len(fields)):
        new_document = {}
        new_document[fields[i]] = documents[i]
        new_documents.append(new_document)
    T_topic_docs_dict = {}
    T_topic_docs_list = []
    doc_num = []
    for i in range(n_clusters):
        keydocs_vector_list_sort = []
        keydocs_vector_list_dist = []
        for k in range(len(doc_v)):
            keydocs_vector_list_dist.append(np.linalg.norm(prj_topic_vector_list[i] - np.array(doc_v[k])))
            keydocs_vector_list_sort.append(np.linalg.norm(prj_topic_vector_list[i] - np.array(doc_v[k])))
        keydocs_vector_list_sort.sort(reverse=False)
        keydocs_list = []
        for j in range(n_clusters):
            keydocs_number = keydocs_vector_list_dist.index(keydocs_vector_list_sort[j])
            keydocs_list.append(new_documents[keydocs_number])

        T_topic_docs_list.append(keydocs_list)
        T_topic_docs_dict[f'topic {i}'] = keydocs_list

    return T_topic_docs_list, T_topic_docs_dict



def clustering(n_clusters, path, save_path, country_list):
    print(f'cluster num = {n_clusters}')

    concat_doc_vector = torch.tensor([])
    word_v = torch.tensor([])
    word_list = np.array([])
    for country in country_list:
        try:
            country_data = pd.read_csv(path + f'tokenized_{country}_data.csv', encoding='utf-8-sig')
            globals()[country + '_document'] = make_document(country_data)
            globals()[country + '_data'] = pd.read_csv(path + f'tokenized_{country}_data.csv',encoding='utf-8-sig')  # 없으면 밑에 실행 안되게끔
            globals()[country + '_word_list'] = np.load(path + f'top_word_list_{country}.npy')
            globals()[country + '_doc_v'] = torch.tensor(np.load(path + f'prj_doc_{country}.npy'))
            globals()[country + '_word_v'] = torch.tensor(np.load(path + f'prj_word_{country}.npy'))
            globals()[country + '_fields'] = make_fields(country_data)
            word_v = torch.concat([word_v, globals()[country + '_word_v']], axis=0)
            word_list = np.concatenate((word_list, globals()[country + '_word_list']), axis=0)
            concat_doc_vector = torch.concat([concat_doc_vector, globals()[country + '_doc_v']], axis=0)

        except ValueError:
            pass
    prj_topic_vector_list = k_means_clustering(n_clusters, concat_doc_vector)
    total_doc_num = []
    for country in country_list:
        globals()[country + '_topic_words_list'], globals()[country + '_topic_words_eucl_dict'], globals()[
            country + '_topic_words_dict'] = Topic_words(globals()[country + '_word_v'],
                                                         globals()[country + '_word_list'], prj_topic_vector_list,
                                                         n_clusters)
        globals()[country + '_topic_docs_list'], globals()[country + '_topic_docs_dict'] = Topic_docs(
            globals()[country + '_doc_v'], globals()[country + '_document'], prj_topic_vector_list,
            globals()[country + '_fields'], n_clusters)
        globals()[country + '_doc_num'] = Document_number(n_clusters, country, globals()[country + '_doc_v'], prj_topic_vector_list)
        total_doc_num = total_doc_num + globals()[country + '_doc_num']

    # Treemap
    print('--------Tree_map 생성중--------')
    doc_count = Counter(total_doc_num).most_common()
    aa = []
    bb = []
    for i in range(len(doc_count)):
        aa.append(doc_count[i][0])
        bb.append(doc_count[i][1])
    doc_mount_list = []

    for i in range(n_clusters):
        if i in aa:
            idx = aa.index(i)
        doc_mount_list.append(bb[idx])
    ratio_list = []
    for i in range(len(doc_mount_list)):
        ratio_list.append(doc_mount_list[i] / sum(doc_mount_list))
    total_doc_ratio = []
    for i in range(n_clusters):
        for j in range(25):
            total_doc_ratio.append(ratio_list[i])

    total_words_list = []
    top_eucli_list = []
    for i in range(n_clusters):
        eucli_list = []
        total_words_list += korean_topic_words_dict[f'topic {i}']
        total_words_list += english_topic_words_dict[f'topic {i}']
        total_words_list += german_topic_words_dict[f'topic {i}']
        total_words_list += chinese_topic_words_dict[f'topic {i}']
        total_words_list += japanese_topic_words_dict[f'topic {i}']

        eucli_list += korean_topic_words_eucl_dict[f'topic {i}']
        eucli_list += english_topic_words_eucl_dict[f'topic {i}']
        eucli_list += german_topic_words_eucl_dict[f'topic {i}']
        eucli_list += chinese_topic_words_eucl_dict[f'topic {i}']
        eucli_list += japanese_topic_words_eucl_dict[f'topic {i}']
        top_eucli_list.append(eucli_list)
    ratio_eucli_list = []
    for i in range(n_clusters):
        X = np.array(top_eucli_list[i])
        exp_a = np.exp(X)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        ratio_eucli_list += list(y)
    topic = []
    for i in range(n_clusters):
        for j in range(25):
            topic.append(f'Topic_{i}')
    treemap_dict = {"topic": topic, "전체문서에서 topic의 비중": total_doc_ratio, "topic 해당하는 단어": total_words_list,
                    "비율": ratio_eucli_list}
    topic_label = pd.DataFrame(treemap_dict)
    topic_label['값'] = topic_label['비율'] * topic_label['전체문서에서 topic의 비중']
    fig = px.treemap(topic_label, path=['topic', 'topic 해당하는 단어'], values='값', title='LGD treepmap')
    fig.update_traces(textposition='middle center', textinfo='percent root+label')
    fig.show()
    fig.write_html(save_path + 'treemap.html')

    # Excel
    print('-------Final_table 생성중-------')
    writer = pd.ExcelWriter(save_path + 'Topic_check_multi.xlsx', engine='xlsxwriter')

    for m in range(n_clusters):
        topic_num = m
        topic_dict = {}
        for country in country_list:
            globals()[country + '_topic_fields'] = []
            globals()[country + '_topic_words'] = []
        topic_num_list = []
        topic_fields_list = []
        for i in range(10):
            topic_num_list.append('')
            topic_fields_list.append('')
        topic_fields_list1 = []
        for country in country_list:
            for i in range(10):
                globals()[country + '_topic_words'].append('')
                globals()[country + '_topic_fields'].append(
                    list(globals()[country + '_topic_docs_list'][topic_num][i].keys())[0])
            topic_fields_list1 = topic_fields_list1 + globals()[country + '_topic_fields']
        count_items = Counter(topic_fields_list1)
        max_c = count_items.most_common(n=2)
        if max_c[0][0] == 'None':
            topic_fields_list[0] = max_c[1][0]
        else:
            topic_fields_list[0] = max_c[0][0]
        topic_fields_list[5] = str('← 한글/영어/중국어/일본어/독일어 각 10개 인접문서의 분야들 중 제일 개수가 많은 분야')

        for country in country_list:
            globals()[country + '_topic_words'][0] = str(globals()[country + '_topic_words_list'][topic_num][:10])

        topic_dict[f'Topic_num_{m}'] = topic_num_list
        topic_dict['토픽 대표 분야'] = topic_fields_list
        for country in country_list:
            topic_dict[f'Top10 {country} Keywords'] = globals()[country + '_topic_words']

        for country in country_list:
            for j in range(2):
                topic_docs_list = []
                for k in range(10):
                    topic_docs_list.append('')

                topic_docs_list[0] = list(globals()[country + '_topic_docs_list'][topic_num][j].values())[0]
                topic_dict[f'{country} Keypatents_{j}'] = topic_docs_list

        aa = pd.DataFrame(topic_dict)
        ab = aa.transpose()
        ab.to_excel(writer, sheet_name=f'Topic_{m}', encoding='utf-8-sig')
    writer.save()
    print('완료')

if __name__ == "__main__":
    clustering(30)