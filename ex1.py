import gc
import sys
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from sklearn.cluster import AgglomerativeClustering

sc = SparkContext(appName="Assignment2_E1")

# Auxiliar function to read the file in chunks
def process_batch(batch):
    global i
    chunk = np.array(batch)
    if i:
        # removing the header
        chunk = chunk[4:, :]
        chunk = chunk.astype(float)
        i = False
        start_bfr(chunk)
    else:
        chunk = chunk.astype(float)
        continue_bfr(chunk)
    return


# Auxiliar functions
def centroid(stats):
    # centroid = SUM / N
    return stats[1] / stats[0]


def variance(stats):
    # variance = (SUMSQ / N) - np.square(SUM / N)
    return stats[2] / stats[0] - np.square((stats[1] / stats[0]))


def calculate_malahanobis(point, centroid, std_dev):
    return np.sqrt(np.sum(np.square((point-centroid)/std_dev)))

# Step 1


def start_bfr(initial_chunk):
    global clusters_indices
    global summarized_ds

    # removing the index before the clustering
    points = initial_chunk[:, 1:]
    initial_clusters = AgglomerativeClustering(
        n_clusters=num_clusters).fit(points)
    # indices of the points in each cluster

    # extracting the labels
    labels = np.array([[x] for x in initial_clusters.labels_])
    points = np.append(points, labels, axis=1)
    aux = np.append(initial_chunk, labels, axis=1)

    clusters_indices = {cluster: [x[0] for x in aux if x[-1] == cluster]
                        for cluster in range(initial_clusters.n_clusters_)}

    clusters = {cluster: points[np.where(initial_clusters.labels_ == cluster)][:, :-1]
                for cluster in range(initial_clusters.n_clusters_)}
    #clusters = {cluster: [x[:-1] for x in points if x[-1] == cluster] for cluster in range(initial_clusters.n_clusters_)}
    del aux
    gc.collect()

    summarized_ds = calc_ds_stats(clusters)

    return


def calc_ds_stats(cluster):
    summarized_clusters = {}
    for cluster_id, points in cluster.items():
        N = len(points)
        SUM = np.sum(points, axis=0)
        SUMSQ = np.sum(np.square(points), axis=0)

        summarized_clusters[cluster_id] = (N, SUM, SUMSQ)

    return summarized_clusters


def continue_bfr(chunk):
    global DISCARD_SET
    global RETAINED_SET
    global COMPRESSION_SET

    batch = sc.parallelize(chunk)
    DISCARD_SET = batch.map(lambda point: point if
                            min([calculate_malahanobis(point[1:], centroid(x[1]), np.sqrt(
                                variance(x[1]), )) for x in summarized_ds.items()])
                            < (2*np.sqrt(518)) else '') \
        .filter(lambda point: point != '') \
        .collect()

    if len(DISCARD_SET) > 0:
        RETAINED_SET = np.append(RETAINED_SET, chunk[np.isin(
            chunk[:, 0], np.array(DISCARD_SET)[:, 0], invert=True)], axis=0)
    else:
        RETAINED_SET = np.append(RETAINED_SET, chunk, axis=0)

    DISCARD_SET = np.array(DISCARD_SET)
    find_cluster(DISCARD_SET)
    DISCARD_SET = []

    if RETAINED_SET.size != 0:
        process_retained()


# step 3
def find_cluster(set):
    global summarized_ds
    global clusters_indices

    # iterating through the chunk
    for point in set:
        min_distance = np.inf

        # calculating the minimum distance to a cluster
        for cluster, stats in summarized_ds.items():
            std = np.square(variance(stats))
            distance = calculate_malahanobis(point[1:], centroid(stats), std)
            if distance < min_distance:
                distance = min_distance
                label = cluster

        # saving the point id to the clusters dictionary
        # dont need to check distance because we already know it is smaller than 2*sqrt(dimension)
        # if it is during finalize() that condition also doesn't apply
        clusters_indices[label].append(point[0])

        # updating the statistics
        # statistics = (N, SUM, SUMSQ)
        # using point[1:] in order to remove the id from the point
        statistics = summarized_ds[label]
        N = statistics[0] + 1
        SUM = statistics[1] + point[1:]
        SUMSQ = statistics[2] + np.square(point[1:])
        summarized_ds.update({label: (N, SUM, SUMSQ)})

    return


def process_retained():
    global RETAINED_SET
    global COMPRESSION_SET

    X = RETAINED_SET.astype(float)
    points = X[:, 1:]
    # distance threshold = 2x number of dimensions(517)
    cs_clusters = AgglomerativeClustering(
        n_clusters=None, distance_threshold=2*np.sqrt(518)).fit(points)

    labels = np.array([[x] for x in cs_clusters.labels_])
    points = np.append(X, labels, axis=1)
    clusters_idx = [[x[:-1] for x in points if x[-1] == cluster]
                    for cluster in set(cs_clusters.labels_)]

    COMPRESSION_SET = [x for x in clusters_idx if len(x) > 1]
    COMPRESSION_SET = np.array(COMPRESSION_SET)

    idx = [x[i][0] for x in clusters_idx if len(x) > 1 for i in range(len(x))]
    RETAINED_SET = np.delete(RETAINED_SET, np.where(
        np.isin(RETAINED_SET[:, 0], idx)), axis=0)

    process_cs()
    return


def process_cs():
    global COMPRESSION_SET
    global summarized_ds
    global clusters_indices

    for clust in COMPRESSION_SET:
        clust = np.array(clust)
        c_summary = (len(clust), np.sum(clust[:, 1:], axis=0), np.sum(
            np.square(clust[:, 1:]), axis=0))
        new_stats = [0, 0, 0]
        var = np.full((518,), np.inf)

        for c, stats in summarized_ds.items():

            new_stats = [stats[0]+c_summary[0], stats[1] +
                         c_summary[1], stats[2]+c_summary[2]]
            n_var = variance(new_stats)

            if all(n_var < var):
                var = n_var
                label = c

        if all(var < 1.1 * 518):
            clusters_indices[label].extend(clust[:, 0])
            summarized_ds.update(
                {label: (new_stats[0], new_stats[1], new_stats[2])})
            COMPRESSION_SET = np.delete(COMPRESSION_SET, np.where(
                np.isin(COMPRESSION_SET, clust)), axis=0)

    return


def finalize():
    global summarized_ds
    global COMPRESSION_SET
    global RETAINED_SET

    final = sc.parallelize(RETAINED_SET)

    final = final.map(lambda point: (min([(calculate_malahanobis(point[1:], centroid(x[1]), np.sqrt(variance(x[1]))), x[0]) for x in summarized_ds.items()], key=lambda i: i[0])[1], {point[0]})) \
        .reduceByKey(lambda a, b: a | b) \
        .map(lambda x: (x[0], list(x[1]))) \
        .collect()

    find_cluster(RETAINED_SET)

    for clust in COMPRESSION_SET:
        clust = np.array(clust)
        c_summary = (len(clust), np.sum(clust[:, 1:], axis=0), np.sum(
            np.square(clust[:, 1:]), axis=0))
        new_stats = [0, 0, 0]
        var = np.full((518,), np.inf)

        for c, stats in summarized_ds.items():
            new_stats = [stats[0]+c_summary[0], stats[1] +
                         c_summary[1], stats[2]+c_summary[2]]
            n_var = variance(new_stats)
            if all(n_var < var):
                var = n_var
                label = c

        clusters_indices[label].extend(clust[:, 0])
        summarized_ds.update(
            {label: (new_stats[0], new_stats[1], new_stats[2])})

    COMPRESSION_SET = []
    RETAINED_SET = []

    return

if __name__ == '__main__':
    file = "data/features.csv"
    try:
        file = sys.argv[1]
    except:
        print("Usage: spark-submit ex2.py <file>")
        exit(1)

    DISCARD_SET = []
    COMPRESSION_SET = []
    RETAINED_SET = np.empty([0, 519])
    summarized_ds = {}
    clusters_indices = {}
    num_clusters = 8
    i = True
    with open('data/features.csv') as f:
        batch = []
        for line in f:
            batch.append(line.rstrip('\n').split(','))
            if len(batch) == 8000:
                process_batch(batch)
                batch = []
    if batch:
        process_batch(batch)

    finalize()
    
    with open('results.txt', 'w') as f:
        for k,v in clusters_indices.items():
            f.write(k + ' : ' + v)
