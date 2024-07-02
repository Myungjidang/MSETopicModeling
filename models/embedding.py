import os
import json
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from tqdm import tqdm
from gensim import corpora
import gensim
import pandas as pd
import re
from numpy import inf
from collections import Counter
from models.utils.MyTop import Top2Vec
import numpy as np


def make_document(country_data):
    country_documents = []
    country_tokens = []
    for i in tqdm(range(len(country_data))):
        country_documents.append(country_data['Documents'][i])
        country_tokens.append(country_data['Tokens'][i])
    return country_documents, country_tokens


def top2vec_embedding(path, country, country_tokens):
    print(f'{country} Embedding')
    model_T = Top2Vec(country_tokens, speed="learn", workers=8)
    topic_words, word_scores, topic_nums = model_T.get_topics()
    country_wv = model_T._get_word_vectors()

    idx2word = model_T.model.wv.index2word
    word2idx = dict()
    for i, word in enumerate(idx2word):
        word2idx[word] = i

    word_list = np.array(idx2word)
    np.save(path + f'top_word_list_{country}.npy', word_list)
    word_vector = country_wv
    np.save(path + f'top_word_{country}.npy', word_vector)
    doc_vector = model_T._get_document_vectors()
    np.save(path + f'top_doc_{country}.npy', doc_vector)
    return word_list, word_vector, doc_vector


def sbert_embedding(path, documents, country_list, country_length):
    print('sbert embedding 오래걸립니다.')
    from sentence_transformers import SentenceTransformer
    s_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = s_model.encode(documents)
    np.save(path + f'sbert_doc.npy', embeddings)
    count = 0
    count_1 = 0
    for i in range(len(country_list)):
        count_1 += country_length[i]
        print(count_1)
        country_embedding = embeddings[int(count):int(count_1)]
        count = count_1
        np.save(path + f'sbert_doc_{country_list[i]}.npy', country_embedding)
    return embeddings


def embedding(path, country_list):
    all_documents = []
    country_length = []
    for country in tqdm(country_list):
        print('---------', country, '---------')
        try:
            country_data = pd.read_csv(path + f'tokenized_{country}_data.csv', encoding='utf-8-sig')  # 없으면 밑에 실행 안되게끔
            country_document, country_token = make_document(country_data)
            country_length.append(len(country_document))
            word_list, word_vector, doc_vector = top2vec_embedding(path, country, country_token)
        except ValueError:
            pass
        all_documents = all_documents + country_document
    s_embedding = sbert_embedding(path, all_documents, country_list, country_length)


if __name__ == "__main__":
    embedding()