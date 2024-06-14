# Algorithm ：Jeffries Matusita distance criterion constrained K-means clustering algorithm


## Overview
This algorithm processes preprocessed LiDAR data to classify vegetation into stressed and non-stressed categories based on clustering and JM distance calculations.

## Input
- **Preprocessed LiDAR Data**: The input data must be preprocessed and ready for analysis.

## Method
1. **Cluster the Data**: Use the "Elbow" method to determine the optimal number of clusters \(K\) and cluster the data into \(K\) classes.
2. **Sort the Clusters**: Sort the clusters in descending order based on the H95 attribute of the cluster centroids. The clusters are denoted as Cluster1 to ClusterK.
3. **Designate Healthy Vegetation Cluster**: Designate Cluster1, which has the highest H95 attribute, as the healthy vegetation cluster. Classify its pixels as "non-stressed pixels".
4. **Iterate Over Other Clusters**: For each cluster from n = 2 to K:
   - Calculate the JM distance value \(D\) between the samples of Cluster1 and Cluster_n.
   - **Classify Pixels Based on JM Distance**:
     - If \(D \geq 1.8\): Classify the pixels of Cluster_n as "suspected stressed pixels".
     - Else: Classify the pixels of Cluster_n as "non-stressed pixels".
5. **End Loop**: Repeat the classification for all clusters.

## Output
- **Pixel-level Binary Classification Map**: The output is a map where each pixel is classified as either stressed or non-stressed based on the specified criteria.
