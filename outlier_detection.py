import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import pandas as pd

#load training and test data
array = np.load("path to n-dimensional array, in numpy array format")
predict_array = np.load("path to data to predict")

def return_PCA(array, number_of_components):        #Return specified number of Principal Components of array

    pca = PCA(n_components=number_of_components)
    data = pca.fit_transform(array)
    # standardize these 2 new features
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)

    return(np_scaled)

def k_means(data):      #Fit k-means for clusters in range(1,20). You can then choose number of clusters based on the score graph

    number_of_clusters = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(data) for i in number_of_clusters]
    scores = [kmeans[i].score(data) for i in range(len(kmeans))]
    fig, ax = plt.subplots()
    ax.plot(number_of_clusters, scores)
    #plt.show()

    cluster = pd.Series(kmeans[2].predict(data))
    print(cluster.shape)
    colors = np.random.rand(3)
    print(cluster.value_counts())

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=cluster.apply(lambda x: colors[x]))
    plt.show()


def empirical_covariance(nd_array):
    clf = EllipticEnvelope(support_fraction=0.7, contamination=0.05)
    model_ellipse = clf.fit(nd_array)
    plot_decision_boundary(clf, "Empirical Covariance")     #Only call this function if you have 2 Principal Components
    return (model_ellipse)

def one_class_svm(nd_array):
    svm = OneClassSVM(nu=0.05, gamma=0.1)
    model_svm = svm.fit(nd_array)
    plot_decision_boundary(svm, "One Class SVM")        #Only call this function if you have 2 Principal Componentsts
    return(model_svm)

def isolation_forest(nd_array):
    ilf = IsolationForest(behaviour='new', contamination=0.05, random_state=42)
    model_ilf = ilf.fit(nd_array)
    plot_decision_boundary(ilf, "Isolation Forest")     #Only call this function if you have 2 Principal Components
    return(model_ilf)

def plot_decision_boundary(model, label):

    legend1 = {}
    plt.figure(1)
    xx1, yy1 = np.meshgrid(np.linspace(-5, 10, 500), np.linspace(-5, 10, 500))
    Z1 = model.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    Z1 = Z1.reshape(xx1.shape)
    legend1['Decision Boundary'] = plt.contour(
        xx1, yy1, Z1, levels=[0], linewidths=2, colors='g')
    plt.scatter(data[:, 0], data[:, 1], color='black', alpha=0.25)
    plt.figure(1)  # two clusters
    plt.title("Outlier detection: " + label)
    plt.show()

principal_components = 2        #Number of Principal Components
data = return_PCA(array, principal_components)
print(data.shape)
k_means(data)       #Call K-means function

#Call functions for other algorithms
model_ellipse = empirical_covariance(data)
model_svm = one_class_svm(data)
model_ilf = isolation_forest(data)

y_predicted = model_ellipse.fit_predict(predict_array)      #Predict labels for data to predict

count = 0
for i in range(len(y_predicted)):
    if y_predicted[i] == -1:
        count = count + 1

print(count/len(y_predicted))
