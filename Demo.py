# -*- coding: utf-8 -*-
# @Author: Xinda
# @Date: 2023/08/04

import numpy as np
from sklearn.cluster import KMeans

def jmdist(data, sorted_class_map):
    """
    Calculate Jeffries-Matusita (JM) distances between classes based on the provided class map.
    """
    class_labels = np.unique(sorted_class_map)
    class_no = len(class_labels)
    feat_num = data.shape[2]
    jm_distances = np.zeros((class_no, class_no))

    for i in range(class_no):
        for j in range(class_no):
            if i == j:
                jm_distances[i, j] = 0.0
            else:
                class_indices_i = np.where(sorted_class_map == class_labels[i])
                class_data_i = data[class_indices_i]
                class_indices_j = np.where(sorted_class_map == class_labels[j])
                class_data_j = data[class_indices_j]
                mu_i = np.mean(class_data_i, axis=0)
                mu_j = np.mean(class_data_j, axis=0)
                sigma_i = np.cov(class_data_i, rowvar=False, ddof=1)
                sigma_j = np.cov(class_data_j, rowvar=False, ddof=1)
                sigma_i += np.eye(sigma_i.shape[0]) * 1e-6
                sigma_j += np.eye(sigma_j.shape[0]) * 1e-6
                a = 0.125 * (mu_i - mu_j) @ np.linalg.inv(0.5 * (sigma_i + sigma_j)) @ (mu_i - mu_j).T \
                    + 0.5 * np.log(np.linalg.det(0.5 * (sigma_i + sigma_j)) / np.sqrt(np.linalg.det(sigma_i) * np.linalg.det(sigma_j)))
                jm_distances[i, j] = 2 * (1 - np.exp(-a))
    return jm_distances

def JCKC(data):
    """
    Perform clustering on data and calculate JM distances between clusters.
    """
    reshaped_data = data.reshape(-1, data.shape[-1])
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(reshaped_data)
    labels = kmeans.labels_
    sorted_indices = np.argsort(-kmeans.cluster_centers_[:, 0])
    sorted_labels = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(sorted_indices):
        sorted_labels[labels == old_idx] = new_idx + 1
    sorted_class_map = sorted_labels.reshape(data.shape[0], data.shape[1])
    jm_distances = jmdist(data, sorted_class_map)
    return jm_distances

if __name__ == '__main__':
    # Generate sample data
    np.random.seed(0)
    data = np.random.rand(100, 100, 10)  # 100x100 image, 10 features

    # Call the function to perform clustering and calculate JM distances
    jm_distances = JCKC(data)

    print("JM Distances between clusters:")
    print(jm_distances)
