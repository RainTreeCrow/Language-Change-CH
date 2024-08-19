import itertools
import logging
import pickle
import random

import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

from collections import defaultdict
from deprecated import deprecated
from scipy.spatial.distance import cdist
from tqdm import tqdm
from string import ascii_uppercase
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from transformers import BertTokenizer

SEED = 42

logging.basicConfig(level=logging.INFO)
np.random.seed(SEED)


def best_kmeans(X, max_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return the best K-Means clustering given the data, a range of K values, and a selection criterion.

    :param X: usage matrix (made of usage vectors)
    :param max_range: range within the number of clusters should lie
    :param criterion: K-selection criterion: 'silhouette' or 'calinski'
    :return: best_model: KMeans model (sklearn.cluster.Kmeans) with best clustering
             scores: list of tuples (k, s) indicating the clustering score s obtained using k clusters
    """
    assert criterion in ['silhouette', 'calinski', 'harabasz', 'calinski-harabasz']

    best_model, best_score = None, -1
    scores = []

    for k in max_range:
        if k < X.shape[0]:
            kmeans = KMeans(n_clusters=k, random_state=SEED)
            cluster_labels = kmeans.fit_predict(X)

            if criterion == 'silhouette':
                score = silhouette_score(X, cluster_labels)
            else:
                score = calinski_harabasz_score(X, cluster_labels)

            scores.append((k, score))

            # if two clusterings yield the same score, keep the one that results from a smaller K
            if score > best_score:
                best_model, best_score = kmeans, score

    return best_model, scores


def to_one_hot(y, num_classes=None):
    """
    Transform a list of categorical labels into the list of corresponding one-hot vectors.
    E.g. [2, 3, 1] -> [[0,0,1,0], [0,0,0,1], [0,1,0,0]]

    :param y: N-dimensional array of categorical class labels
    :param num_classes: the number C of distinct labels. Inferred from `y` if unspecified.
    :return: N-by-C one-hot label matrix
    """
    if num_classes:
        K = num_classes
    else:
        K = np.max(y) + 1

    one_hot = np.zeros((len(y), K))

    for i in range(len(y)):
        one_hot[i, y[i]] = 1

    return one_hot


def usage_distribution(predictions, time_labels):
    """
    :param predictions: The clustering predictions
    :param time_labels:
    :return:
    """
    if predictions.ndim > 2:
        raise ValueError('Cluster probabilities has too many dimensions: {}'.format(predictions.ndim))
    if predictions.ndim == 1:
        predictions = to_one_hot(predictions)

    label_set = sorted(list(set(time_labels)))
    t2i = {t: i for i, t in enumerate(label_set)}

    n_clusters = predictions.shape[1]
    n_periods = len(label_set)
    usage_distr = np.zeros((n_clusters, n_periods))

    for pred, t in zip(predictions, time_labels):
        usage_distr[:, t2i[t]] += pred.T

    return usage_distr


def cluster_usages(Uw, k_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return the best clustering model for a usage matrix.

    :param Uw: usage matrix
    :param k_range: range of possible K values (number of clusters)
    :param criterion: K selection criterion; depends on clustering method
    :return: best clustering model
    """
    # standardize usage matrix by removing the mean and scaling to unit variance
    X = preprocessing.StandardScaler().fit_transform(Uw)

    # get best model according to a K-selection criterion
    best_model, _ = best_kmeans(X, k_range, criterion=criterion)

    return best_model


def obtain_clusterings(usages, out_path, k_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return and save dictionary mapping lemmas to their best clustering model, given a criterion.

    :param usages: dictionary mapping lemmas to their tensor data and metadata
    :param out_path: output path to store clustering models
    :param method: K-Means or Gaussian Mixture Model ('kmeans' or 'gmm')
    :param k_range: range of possible K values (number of clusters)
    :param criterion: K selection criterion; depends on clustering method
    :return: dictionary mapping lemmas to their best clustering model
    """
    clusterings = {}  # dictionary mapping lemmas to their best clustering
    for w in tqdm(usages):
        print(w)
        Uw, _, _, _ = usages[w]
        clusterings[w] = cluster_usages(Uw, method, k_range, criterion)

    with open(out_path, 'wb') as f:
        pickle.dump(clusterings, file=f)

    return clusterings


def plot_usage_distribution(usages, clusterings, out_dir, normalized=False):
    """
    Save plots of probability- or frequency-based usage distributions.

    :param usages: dictionary mapping lemmas to their tensor data and metadata
    :param clusterings: dictionary mapping lemmas to their best clustering model
    :param out_dir: output directory for plots
    :param normalized: whether to normalize usage distributions
    """
    for word in clusterings:
        _, _, _, t_labels = usages[word]
        best_model = clusterings[word]

        # create usage distribution based on clustering results
        usage_distr = usage_distribution(best_model.labels_, t_labels)
        if normalized:
            usage_distr = preprocessing.normalize(usage_distr, norm='l1', axis=0)

        # create a bar plot with plotly
        data = []
        for i in range(usage_distr.shape[0]):
            data.insert(0, go.Bar(
                y=usage_distr[i, :],
                name='usage {}'.format(ascii_uppercase[i])
            ))
        layout = go.Layout(title=word,
                           xaxis=dict(
                               ticktext=list(np.arange(1910, 2009, 10)),
                               tickvals=list(np.arange(10))),
                           barmode='stack')

        fig = go.Figure(data=data, layout=layout)
        pio.write_image(fig, '{}/{}_{}.pdf'.format(
            out_dir,
            word,
            'prob' if normalized else 'freq'))


def get_prototypes(word, clustering, usages, n=5, window=10):
    if len(usages) == 4:
        U_w, contexts, positions, t_labels = usages
    else:
        raise ValueError('Invalid argument "usages"')

    # list of placeholder for cluster matrices and the respective sentences
    clusters = []
    snippet_coords = []
    for i in range(clustering.cluster_centers_.shape[0]):
        clusters.append(None)
        snippet_coords.append([])

    # fill placeholders with cluster matrices
    for u_w, sent, pos, cls in zip(U_w, contexts, positions, clustering.predict(U_w)):
        if clusters[cls] is None:
            clusters[cls] = u_w
        else:
            clusters[cls] = np.vstack((clusters[cls], u_w))
        snippet_coords[cls].append((sent, pos))


    prototypes = []
    for cls in range(len(clusters)):
        # each cluster is represented by a list of sentences
        prototypes.append([])

        # skip cluster if it is empty for this interval
        if clusters[cls] is None:
            continue

        # obtain the n closest data points in this cluster to overall cluster centers
        nearest = np.argsort(cdist(clustering.cluster_centers_, clusters[cls]), axis=1)[:, -n:]

        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # loop through cluster-specific indices
        for i in nearest[cls]:
            sent, pos = snippet_coords[cls][i]
            # sent = sent.split()

            sent = tokenizer.convert_ids_to_tokens(sent)
            assert sent[pos] == word
            sent[pos] = '[[{}]]'.format(word)
            sent = sent[pos - window: pos + window + 1]
            sent = ' '.join(sent)

            if sent not in prototypes[-1]:
                prototypes[-1].append(sent)
            if len(prototypes) == n:
                break

    return prototypes