import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot  as plt
import pandas as pd
import pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter


def Orthogonal_projection(epoch, path, country, doc2vec, word2vec, word_list, sbert_doc):
    WHOLE_SIZE = len(doc2vec)
    BATCH_SIZE = len(doc2vec)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = 0.9
    epoches = epoch
    matrix = torch.rand(300, 768, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([matrix], lr=1e-4)
    mse = torch.nn.MSELoss()
    loss1_list = [10000]
    count = 0
    for epoch in tqdm(range(epoches)):

        for i in range(WHOLE_SIZE // BATCH_SIZE):
            input_temp = doc2vec[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].to(device)
            label_temp = sbert_doc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].to(device)

            loss1 = mse(torch.mm(input_temp, matrix), label_temp)
            loss2 = mse(torch.mm(matrix, torch.transpose(matrix, 0, 1)), torch.eye(300, device=device))
            total_loss = loss1 * (1 - alpha) + loss2 * (alpha)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        loss1_list.append(loss1)
        if loss1_list[-2] < loss1_list[-1]:
            count += 1
            if count == 5:
                print("Early stopping")
                print(f'total_loss:{total_loss:<10.7f} loss_Transfer:{loss1:<10.7f} loss_Ortho:{loss2:<10.7f}')
                break
        if epoch % 1000 == 0:
            print(f'epoch:{epoch} total_loss:{total_loss:<10.7f} loss_Transfer:{loss1:<10.5f} loss_Ortho:{loss2:<10.7f}')

    word_vector = torch.mm(word2vec.to(device), matrix)
    doc_vector = torch.mm(doc2vec.to(device), matrix)

    prj_word_v = []
    for i in range(len(word_vector)):
        prj_word_v.append(word_vector[i].detach().cpu().numpy())

    prj_doc_v = []
    for i in range(len(doc_vector)):
        prj_doc_v.append(doc_vector[i].detach().cpu().numpy())
    np.save(path + f'prj_doc_{country}.npy', prj_doc_v)
    np.save(path + f'prj_word_{country}.npy', prj_word_v)
    return prj_doc_v, prj_word_v


def projection(path, country_list):
    epoch = 50000
    all_documents = []
    country_length = []
    for country in tqdm(country_list):
        print('---------', country, '---------')
        try:
            country_data = pd.read_csv(path + f'tokenized_{country}_data.csv', encoding='utf-8-sig')  # 없으면 밑에 실행 안되게끔
            doc2vec = torch.tensor(np.load(path + f'top_doc_{country}.npy'))
            word2vec = torch.tensor(np.load(path + f'top_word_{country}.npy'))
            word_list = np.load(path + f'top_word_list_{country}.npy')
            sbert_doc = torch.tensor(np.load(path + f'sbert_doc_{country}.npy'))
            country_document_vector, country_word_vector = Orthogonal_projection(epoch, path, country, doc2vec, word2vec, word_list, sbert_doc)
        except ValueError:
            pass



if __name__ == "__main__":
    projection(epoch)