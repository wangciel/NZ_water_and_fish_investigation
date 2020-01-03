import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.cluster import KMeans

random.seed(309)
np.random.seed(309)


fish_dir = './fish_output.csv'
df = pd.read_csv(fish_dir)
value,label = df[['seg_phos', 'seg_psize', 'seg_pet', 'seg_mat', 'seg_decs','US_RockPhos', 'USCalcium']],df[['exotic.bio']].astype(bool)

X_train, X_test, y_train, y_test = train_test_split( value, label, test_size=0.33, shuffle=True)
print(len(X_train), len(X_test))

def knn_classification(X_train, X_test, y_train):

    KNN = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree')
    KNN.fit(X_train, np.ravel(y_train) )
    return KNN.predict(X_test)

def mlp_classification(X_train, X_test, y_train):
    MLP = MLPClassifier( max_iter= 2000,  hidden_layer_sizes=(14, ), batch_size= 8)
    MLP.fit(X_train, np.ravel(y_train))
    return MLP.predict(X_test)

def kmeans_clus(X_train, X_test):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
    kmeans_predict = kmeans.predict(X_test)
    return kmeans_predict

if __name__ == '__main__':
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

    plt.savefig('part1_figure.png')

