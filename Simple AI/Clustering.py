import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

def k_mean():
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets.samples_generator import make_blobs

    np.random.seed(0)

    '''
    Next we will be making random clusters of points by using the make_blobs class. The make_blobs class can take in many inputs, but we will be using these specific ones.

    Input
    n_samples: The total number of points equally divided among clusters.
    Value will be: 5000
    centers: The number of centers to generate, or the fixed center locations.
    Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
    cluster_std: The standard deviation of the clusters.
    Value will be: 0.9
    
    Output
    X: Array of shape [n_samples, n_features]. (Feature Matrix)
    The generated samples.
    y: Array of shape [n_samples]. (Response Vector)
    The integer labels for cluster membership of each sample.
    '''

    X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

    plt.scatter(X[:, 0], X[:, 1], marker='.')

    '''
    init: Initialization method of the centroids.
    Value will be: "k-means++"
    k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
    n_clusters: The number of clusters to form as well as the number of centroids to generate.
    Value will be: 4 (since we have 4 centers)
    n_init: Number of time the k-means algorithm will be run with different centroid seeds. 
    The final results will be the best output of n_init consecutive runs in terms of inertia.
    Value will be: 12
    '''

    k_means = KMeans(init="k-means++", n_clusters=3, n_init=12)

    k_means.fit(X)

    k_means_labels = k_means.labels_

    k_means_cluster_centers = k_means.cluster_centers_

    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=(6, 4))

    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

    # Create a plot
    ax = fig.add_subplot(1, 1, 1)

    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
        # Create a list of all data points, where the data poitns that are
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)

        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]

        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    # Title of the plot
    ax.set_title('KMeans')

    # Remove x-axis ticks
    ax.set_xticks(())

    # Remove y-axis ticks
    ax.set_yticks(())

    # Show the plot
    plt.show()


def k_mean2():
    import pandas as pd
    cust_df = pd.read_csv("Cust_Segmentation.csv")
    cust_df.head()

    df = cust_df.drop('Address', axis=1)
    df.head()

    from sklearn.preprocessing import StandardScaler
    X = df.values[:, 1:]
    X = np.nan_to_num(X)
    Clus_dataSet = StandardScaler().fit_transform(X)
    Clus_dataSet

    clusterNum = 3
    k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
    k_means.fit(X)
    labels = k_means.labels_
    print(labels)

    # We can easily check the centroid values by averaging the features in each cluster.

    df.groupby('Clus_km').mean()

    area = np.pi * (X[:, 1]) ** 2
    plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
    plt.xlabel('Age', fontsize=18)
    plt.ylabel('Income', fontsize=16)

    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    # plt.ylabel('Age', fontsize=18)
    # plt.xlabel('Income', fontsize=16)
    # plt.zlabel('Education', fontsize=16)
    ax.set_xlabel('Education')
    ax.set_ylabel('Age')
    ax.set_zlabel('Income')

    ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))


def AgglomerativeHierarchicalClustering():  # agglomerative is the bottom up approach.
    import numpy as np
    import pandas as pd
    from scipy import ndimage
    from scipy.cluster import hierarchy
    from scipy.spatial import distance_matrix
    from matplotlib import pyplot as plt
    from sklearn import manifold, datasets
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets.samples_generator import make_blobs

    X1, y1 = make_blobs(n_samples=50, centers=[[4, 4], [-2, -1], [1, 1], [10, 4]], cluster_std=0.9)

    plt.scatter(X1[:, 0], X1[:, 1], marker='o')

    '''
    The Agglomerative Clustering class will require two inputs:

    n_clusters: The number of clusters to form as well as the number of centroids to generate.
    Value will be: 4
    linkage: Which linkage criterion to use. 
    The linkage criterion determines which distance to use between sets of observation. 
    The algorithm will merge the pairs of cluster that minimize this criterion.
    Value will be: 'complete'
    Note: It is recommended you try everything with 'average' as well
    '''

    agglom = AgglomerativeClustering(n_clusters=4, linkage='complete')

    agglom.fit(X1, y1)

    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(6, 4))

    # These two lines of code are used to scale the data points down,
    # Or else the data points will be scattered very far apart.

    # Create a minimum and maximum range of X1.
    x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

    # Get the average distance for X1.
    X1 = (X1 - x_min) / (x_max - x_min)

    # This loop displays all of the datapoints.
    for i in range(X1.shape[0]):
        # Replace the data points with their respective cluster value
        # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
        plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
                 color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # Remove the x ticks, y ticks, x and y axis
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')

    # Display the plot of the original data before clustering
    plt.scatter(X1[:, 0], X1[:, 1], marker='.')
    # Display the plot
    plt.show()

    dist_matrix = distance_matrix(X1, X1)
    print(dist_matrix)

    Z = hierarchy.linkage(dist_matrix, 'complete')

    dendro = hierarchy.dendrogram(Z)


def AgglomerativeHierarchicalClustering2():
    filename = 'cars_clus.csv'

    # Read csv
    pdf = pd.read_csv(filename)
    print("Shape of dataset: ", pdf.shape)

    pdf.head(5)

    # Data Cleaning
    # lets simply clear the dataset by dropping the rows that have null value:

    print("Shape of dataset before cleaning: ", pdf.size)
    pdf[['sales', 'resale', 'type', 'price', 'engine_s',
         'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
         'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
                                   'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
                                   'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
    pdf = pdf.dropna()
    pdf = pdf.reset_index(drop=True)
    print("Shape of dataset after cleaning: ", pdf.size)
    pdf.head(5)

    featureset = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

    from sklearn.preprocessing import MinMaxScaler
    x = featureset.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    feature_mtx = min_max_scaler.fit_transform(x)
    feature_mtx[0:5]

    # Clustering using Scipy

    import scipy
    leng = feature_mtx.shape[0]
    D = scipy.zeros([leng, leng])
    for i in range(leng):
        for j in range(leng):
            D[i, j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

    import pylab
    import scipy.cluster.hierarchy
    Z = hierarchy.linkage(D, 'complete')

    # Essentially, Hierarchical clustering does not require a pre-specified number of clusters.
    # However, in some applications we want a partition of disjoint clusters just as in flat clustering.
    # So you can use a cutting line:

    from scipy.cluster.hierarchy import fcluster
    max_d = 3
    clusters = fcluster(Z, max_d, criterion='distance')

    # Also, you can determine the number of clusters directly:

    from scipy.cluster.hierarchy import fcluster
    k = 5
    clusters = fcluster(Z, k, criterion='maxclust')


    fig = pylab.figure(figsize=(18, 50))

    def llf(id):
        return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))

    dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')

    # Clustering using scikit-learn

    dist_matrix = distance_matrix(feature_mtx, feature_mtx)
    print(dist_matrix)

    agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
    agglom.fit(feature_mtx)
    agglom.labels_

    pdf['cluster_'] = agglom.labels_
    pdf.head()

    import matplotlib.cm as cm
    n_clusters = max(agglom.labels_) + 1
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    cluster_labels = list(range(0, n_clusters))

    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(16, 14))

    for color, label in zip(colors, cluster_labels):
        subset = pdf[pdf.cluster_ == label]
        for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), rotation=25)
        plt.scatter(subset.horsepow, subset.mpg, s=subset.price * 10, c=color, label='cluster' + str(label), alpha=0.5)
    #    plt.scatter(subset.horsepow, subset.mpg)
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')

    pdf.groupby(['cluster_', 'type'])['cluster_'].count()

    agg_cars = pdf.groupby(['cluster_', 'type'])['horsepow', 'engine_s', 'mpg', 'price'].mean()

    plt.figure(figsize=(16, 10))
    for color, label in zip(colors, cluster_labels):
        subset = agg_cars.loc[(label,),]
        for i in subset.index:
            plt.text(subset.loc[i][0] + 5, subset.loc[i][2],
                     'type=' + str(int(i)) + ', price=' + str(int(subset.loc[i][3])) + 'k')
        plt.scatter(subset.horsepow, subset.mpg, s=subset.price * 20, c=color, label='cluster' + str(label))
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')


def DensityBasedClustering():
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    '''
    Data generation¶
    The function below will generate the data points and requires these inputs:
    
    centroidLocation: Coordinates of the centroids that will generate the random data.
    Example: input: [[4,3], [2,-1], [-1,4]]
    numSamples: The number of data points we want generated, 
    split over the number of centroids (# of centroids defined in centroidLocation)
    Example: 1500
    clusterDeviation: The standard deviation between the clusters. The larger the number, the further the spacing.
    Example: 0.5
    '''

    def createDataPoints(centroidLocation, numSamples, clusterDeviation):
        # Create random data and store in feature matrix X and response vector y.
        X, y = make_blobs(n_samples=numSamples, centers=centroidLocation,
                          cluster_std=clusterDeviation)

        # Standardize features by removing the mean and scaling to unit variance
        X = StandardScaler().fit_transform(X)
        return X, y

    X, y = createDataPoints([[4, 3], [2, -1], [-1, 4]], 1500, 0.5)

    '''
    Modeling¶
    DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. 
    This technique is one of the most common clustering algorithms which works based on density of object. 
    The whole idea is that if a particular point belongs to a cluster, 
    it should be near to lots of other points in that cluster.
    
    It works based on two parameters: Epsilon and Minimum Points
    Epsilon determine a specified radius that if includes enough number of points within, we call it dense area
    minimumSamples determine the minimum number of data points we want in a neighborhood to define a cluster.
    '''

    epsilon = 0.3
    minimumSamples = 7
    db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
    labels = db.labels_

    '''
    Distinguish outliers
    Lets Replace all elements with 'True' in core_samples_mask that are in the cluster, 
    'False' if the points are outliers.
    '''

    # Firts, create an array of booleans using the labels from db.
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


    # Remove repetition in labels by turning it into a set.
    unique_labels = set(labels)


    '''
    Data visualization
    '''
    # Create colors for the clusters.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Plot the points with colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        # Plot the datapoints that are clustered
        xy = X[class_member_mask & core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)

        # Plot the outliers
        xy = X[class_member_mask & ~core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)


def DensityBasedClustering2():   # Weather Station Clustering using DBSCAN & scikit-learn
    import csv
    import pandas as pd
    import numpy as np

    filename = 'weather-stations20140101-20141231.csv'

    # Read csv
    pdf = pd.read_csv(filename)
    pdf.head(5)

    '''
    3-Cleaning
    Lets remove rows that dont have any value in the Tm field.
    '''

    pdf = pdf[pd.notnull(pdf["Tm"])]
    pdf = pdf.reset_index(drop=True)
    pdf.head(5)

    # Visualization
    '''
    Visualization of stations on map using basemap package. 
    The matplotlib basemap toolkit is a library for plotting 2D data on maps in Python. 
    Basemap does not do any plotting on it’s own, 
    but provides the facilities to transform coordinates to a map projections.
    '''


    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from pylab import rcParams

    rcParams['figure.figsize'] = (14, 10)

    llon = -140
    ulon = -50
    llat = 40
    ulat = 65

    pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) & (pdf['Lat'] < ulat)]

    my_map = Basemap(projection='merc',
                     resolution='l', area_thresh=1000.0,
                     llcrnrlon=llon, llcrnrlat=llat,  # min longitude (llcrnrlon) and latitude (llcrnrlat)
                     urcrnrlon=ulon, urcrnrlat=ulat)  # max longitude (urcrnrlon) and latitude (urcrnrlat)

    my_map.drawcoastlines()
    my_map.drawcountries()
    # my_map.drawmapboundary()
    my_map.fillcontinents(color='white', alpha=0.3)
    my_map.shadedrelief()

    # To collect data based on stations

    xs, ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
    pdf['xm'] = xs.tolist()
    pdf['ym'] = ys.tolist()

    # Visualization1
    for index, row in pdf.iterrows():
        #   x,y = my_map(row.Long, row.Lat)
        my_map.plot(row.xm, row.ym, markerfacecolor=([1, 0, 0]), marker='o', markersize=5, alpha=0.75)
    # plt.text(x,y,stn)
    plt.show()

    '''
    Clustering of stations based on their location i.e. Lat & Lon
    '''
    from sklearn.cluster import DBSCAN
    import sklearn.utils
    from sklearn.preprocessing import StandardScaler
    sklearn.utils.check_random_state(1000)
    Clus_dataSet = pdf[['xm', 'ym']]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

    # Compute DBSCAN
    db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    pdf["Clus_Db"] = labels

    realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels))

    # A sample of clusters
    pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5)

    set(labels)

    '''
    Visualization of clusters based on location
    '''

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from pylab import rcParams

    rcParams['figure.figsize'] = (14, 10)

    my_map = Basemap(projection='merc',
                     resolution='l', area_thresh=1000.0,
                     llcrnrlon=llon, llcrnrlat=llat,  # min longitude (llcrnrlon) and latitude (llcrnrlat)
                     urcrnrlon=ulon, urcrnrlat=ulat)  # max longitude (urcrnrlon) and latitude (urcrnrlat)

    my_map.drawcoastlines()
    my_map.drawcountries()
    # my_map.drawmapboundary()
    my_map.fillcontinents(color='white', alpha=0.3)
    my_map.shadedrelief()

    # To create a color map
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

    # Visualization1
    for clust_number in set(labels):
        c = (([0.4, 0.4, 0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = pdf[pdf.Clus_Db == clust_number]
        my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker='o', s=20, alpha=0.85)
        if clust_number != -1:
            cenx = np.mean(clust_set.xm)
            ceny = np.mean(clust_set.ym)
            plt.text(cenx, ceny, str(clust_number), fontsize=25, color='red', )
            print("Cluster " + str(clust_number) + ', Avg Temp: ' + str(np.mean(clust_set.Tm)))

    '''
    Clustering of stations based on their location, mean, max, and min Temperature
    '''

    from sklearn.cluster import DBSCAN
    import sklearn.utils
    from sklearn.preprocessing import StandardScaler
    sklearn.utils.check_random_state(1000)
    Clus_dataSet = pdf[['xm', 'ym', 'Tx', 'Tm', 'Tn']]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    pdf["Clus_Db"] = labels

    realClusterNum = len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels))

    # A sample of clusters
    pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5)

    '''
    Visualization of clusters based on location and Temperature
    '''

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from pylab import rcParams
    % matplotlib
    inline
    rcParams['figure.figsize'] = (14, 10)

    my_map = Basemap(projection='merc',
                     resolution='l', area_thresh=1000.0,
                     llcrnrlon=llon, llcrnrlat=llat,  # min longitude (llcrnrlon) and latitude (llcrnrlat)
                     urcrnrlon=ulon, urcrnrlat=ulat)  # max longitude (urcrnrlon) and latitude (urcrnrlat)

    my_map.drawcoastlines()
    my_map.drawcountries()
    # my_map.drawmapboundary()
    my_map.fillcontinents(color='white', alpha=0.3)
    my_map.shadedrelief()

    # To create a color map
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

    # Visualization1
    for clust_number in set(labels):
        c = (([0.4, 0.4, 0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = pdf[pdf.Clus_Db == clust_number]
        my_map.scatter(clust_set.xm, clust_set.ym, color=c, marker='o', s=20, alpha=0.85)
        if clust_number != -1:
            cenx = np.mean(clust_set.xm)
            ceny = np.mean(clust_set.ym)
            plt.text(cenx, ceny, str(clust_number), fontsize=25, color='red', )
            print("Cluster " + str(clust_number) + ', Avg Temp: ' + str(np.mean(clust_set.Tm)))


if __name__ == '__main__':
    k_mean()
    k_mean2()

