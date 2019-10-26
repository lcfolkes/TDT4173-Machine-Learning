import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
import matplotlib.cm as cm

def get_class_data():
    #format of data
    #dataset = {'0':[[1,2],[2,3],[3,1]], '1':[[6,5],[7,7],[8,6]]}
    #new_features = [5,7]

    df = pd.read_csv('./dataset/knn_classification.csv', sep=',')
    df2 = df.drop([123])  #drop row 124th row

    classes = df2.y.unique()

    dataset = {c:[] for c in classes}
    for i, row in df2.iterrows():
        c = row['y']
        dataset[c].append(list(row[:-1]))

    #for i in dataset:
    #   for ii in dataset[i]:
    #    plt.scatter(ii[0],ii[1], s=100,color=cm.hot(i*100))
    #plt.show()

    return df, dataset

def get_reg_data():
    df = pd.read_csv('./dataset/knn_regression.csv', sep=',')
    df2 = df.drop([123])
    dataset = np.array(df2)
    return df, dataset

def knn_class(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    #get group of k nearest distances
    neighbors = [i[1] for i in sorted(distances)[:k]]
    #print([i[0] for i in sorted(distances)[:k]])
    vote_result = Counter(neighbors).most_common(1)[0][0]
    return vote_result, neighbors

def knn_reg(data,predict, k=2):
    #if len(data) >= k:
    #    warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for row in data:
        features = row[:-1]
        y = row[-1]
        euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
        distances.append([euclidean_distance, y])

    #mean of y values of k nearest neighbors
    neighbors = [i[1] for i in sorted(distances)[:k]]
    result = np.round(np.array(neighbors).mean(),2)
    return result, neighbors


### Run Code ###

### Classification
dataset_class = get_class_data()
predict_class = [6.3, 2.7, 4.91, 1.8]
result_class = knn_class(dataset_class[1], predict_class, k=10)
print("### Classification ###")
print("k=10 \nPredicted class of {} is {}, with k nearest neighbors having classes {}. \nOriginal array is: \n{}".
      format(predict_class,result_class[0], result_class[1], dataset_class[0].iloc[123]))

print()

### Regression
dataset_reg = get_reg_data()
predict_reg = [6.3, 2.7, 4.91]
result_reg = knn_reg(dataset_reg[1], predict_reg, k=10)
print("### Regression ###")
print("k=10 \nPredicted y value of {} is {}, with k nearest neighbors having y values {}. \nOriginal array is: \n{}".
      format(predict_reg,result_reg[0], result_reg[1], dataset_reg[0].iloc[123]))
