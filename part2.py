import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif

random.seed(309)
np.random.seed(309)

def parse_water_fish_data():

    fish_dir = './fish_output.csv'
    fish_df = pd.read_csv(fish_dir)

    fish_dir = './water_output.csv'
    water_df = pd.read_csv(fish_dir)
    water_df = water_df.rename(columns= {"river":"y", "sdate": "locality"})


    matching_fdval = []

    for i in range(0, len(fish_df)):
        river, year = fish_df['locality'][i], fish_df['y'][i]

        fdval_value = 0.001
        for idx in range(0, len(fish_df)):

            if(river == water_df['locality'][idx] and year == water_df['y'][idx]):
                fdval_value = water_df['fdval'][idx]
                continue
        matching_fdval.append(fdval_value)

    print(matching_fdval, len(matching_fdval))
    fish_df['fdval'] = matching_fdval
    fish_df.to_csv("fish_water_output.csv",index=False)
    print(fish_df)

def knn_classification(X_train, X_test, y_train):

    KNN = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree')
    KNN.fit(X_train, np.ravel(y_train) )
    return KNN.predict(X_test)

def mlp_classification(X_train, X_test, y_train):
    #16 200 are better for before feature selection
    #10 8 are better for after feature selection
    MLP = MLPClassifier( max_iter= 2000,  hidden_layer_sizes=(10, ), batch_size= 8)
    MLP.fit(X_train, np.ravel(y_train))
    return MLP.predict(X_test)


def kmeans_clus(X_train, X_test):
    kmeans = KMeans(n_clusters=2, init='random',max_iter=200, n_init=2 , random_state=0).fit(X_train)
    kmeans_predict = kmeans.predict(X_test)
    return kmeans_predict


def classification_clustering(X_train, X_test, y_train, y_test, graph_title):
    knn_predict, mlp_predict = knn_classification(X_train, X_test, y_train), mlp_classification(X_train, X_test, y_train)
    knn_acc, mlp_acc = metrics.accuracy_score(knn_predict,y_test), metrics.accuracy_score(mlp_predict,y_test)

    knn_fpr, knn_tpr, knn_thresholds = metrics.roc_curve(y_test, knn_predict)
    knn_auc = metrics.auc(knn_fpr,knn_tpr)

    mlp_fpr, mlp_tpr, mlp_thresholds = metrics.roc_curve(y_test, mlp_predict)
    mlp_auc = metrics.auc(mlp_fpr,mlp_tpr)

    print("acc, ", knn_acc, " ", mlp_acc)
    print("auc, ", knn_auc, " ", mlp_auc)

    kmeans_predict = kmeans_clus(X_train,X_test)
    kmeans_fpr, kmeans_tpr, kmeans_thresholds = metrics.roc_curve(y_test, kmeans_predict)
    kmeans_auc = metrics.auc(kmeans_fpr,kmeans_tpr)
    print ("kmeans f1, " ,metrics.f1_score(y_test, kmeans_predict) )

    plt.figure()
    plt.plot(knn_fpr, knn_tpr, color='darkorange',label='KNN ROC curve (area = %0.2f)' % knn_auc)
    plt.plot(mlp_fpr, mlp_tpr, color='navy',label='MLP ROC curve (area = %0.2f)' % mlp_auc)
    plt.plot(kmeans_fpr, kmeans_tpr, color='aqua',label='Kmeans ROC curve (area = %0.2f)' % kmeans_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(graph_title)

def find_top5_by_score(scores,labels):
    attnames,line = [],''
    for score, attname in sorted(zip(scores, labels), reverse=True)[:5]:
        attnames.append(attname)

    return attnames

if __name__ == '__main__':

    fish_dir = './fish_water_output.csv'
    df = pd.read_csv(fish_dir)
    categories = ['seg_phos', 'seg_psize', 'seg_pet', 'seg_mat', 'seg_decs', 'US_RockPhos', 'USCalcium', 'fdval']
    value, label = df[['seg_phos', 'seg_psize', 'seg_pet', 'seg_mat', 'seg_decs', 'US_RockPhos', 'USCalcium', 'fdval']], \
                   df[['exotic.bio']].astype(bool)

    # X_train, X_test, y_train, y_test = train_test_split(value, label, test_size=0.33, shuffle=True)
    # graph_title='part2_before_feature_selection_figure.png'
    # classification_clustering(X_train, X_test, y_train, y_test, graph_title)

    # Infomation Gain to select top 5 features
    info_scores = mutual_info_classif(value,label)
    print(info_scores)
    attribute_names = find_top5_by_score(info_scores, categories)
    print(attribute_names)

    # feature selection data
    X_train, X_test, y_train, y_test = train_test_split(value[attribute_names] , label, test_size=0.33, shuffle=True)
    graph_title = 'part2_after_feature_selection_figure.png'
    classification_clustering(X_train, X_test, y_train, y_test, graph_title)
