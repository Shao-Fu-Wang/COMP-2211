# -*- coding: utf-8 -*-
"""lab4_task.ipynb

Automatically generated by Colaboratory.

# **COMP 2211 Exploring Artificial Intelligence** #
## Lab 4 K-Means Clustering ##
"""

import numpy as np
from scipy import stats


### Task 1: The K-prototype Clustering Algorithm
class KPrototypes:
    def __init__(self, k, X, n_features, max_iter=100):
        """
        Initialization function for KPrototypes class. It is called every time an object is created from this class.
        :param k: int, required
        The number of clusters to form as well as the number of prototypes to generate.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param n_features: int, required
        The number of dimensions of each data instance as well as the number of features. Including both categorical and
        numerical features.
        :param max_iter: int, default=100
        The maximum number of iterations of the K-prototypes algorithm for a single run.
        """
        self.k = k
        self.n_features = n_features
        self.max_iter = max_iter

        # TODO: Randomly select k data points from X as the initial prototypes.
        # Hint: Do not hardcode the indices of selected data points.
        # You may use Numpy's 'random.randint' function to randomly choose the initial prototypes.
        selected_idx =  np.random.randint(X.shape[0], size = k)                                                # ndarray of shape (k, )
        self.prototypes = X[selected_idx]                                               # ndarray of shape (k, n_features)

    def euclidean_distance(self, X, is_categ, debug_prototypes=None):
        """
        Calculate the Euclidean distance between each data point and prototypes. Only consider the numerical feature
        columns of each data point and prototypes.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param is_categ: ndarray of shape (n_features,), required
        Array of boolean values indicates whether each feature column is categorical or not.
        e.g., is_categ=[False, False, True, True] indicates that feature 2 and feature 3 of dataset X are categorical.
        Then, the remaining feature columns (i.e., feature 0 and feature 1) are numerical.
        :param debug_prototypes: ndarray of shape (k, n_features), optional
        If debug_prototypes is given, it will be used in the calculation rather than the stored prototypes in the Class
        Object. This argument is only used to help you test the function independently. Do not use this argument in the
        fit_predict function.

        :return: ndarray of shape (n_prototypes, n_samples)
        The Euclidean distance between each data point and prototypes.
        """
        prototypes = debug_prototypes if debug_prototypes is not None else self.prototypes
        # TODO: only use the numerical feature columns to calculate the Euclidean distance between each data point and
        #       prototypes.
        # Hints:
        #   - Notice that X and prototypes have mismatched shapes.
        #     X: (n_samples, n_features)                    prototypes: (n_prototypes, n_features)
        #   - Only the numerical feature columns are used to calculate the Euclidean distance.
        #     X: (n_samples, n_numerical_features)          prototypes: (n_prototypes, n_numerical_features)
        #   - You may use Numpy's 'count_nonzero', 'reshape', and 'linalg.norm' (or 'sqrt' & 'sum') functions.
        #   - Try broadcasting.
        n_numer_features =  np.sum(is_categ == False)                                                                      # Optional: number of numerical feature columns
        reshaped_numer_prototypes =  np.delete(prototypes, is_categ == True, axis = 1)                                                                # Optional: reshape the numerical prototypes so that we can do broadcasting later
        reshaped_numer_prototypes =  reshaped_numer_prototypes[:, np.newaxis] 
        reshaped_X =  np.delete(X, is_categ == True, axis = 1)                                                                # Optional: reshape the numerical prototypes so that we can do broadcasting later
        # print("reshaped_X shape= ", reshaped_numer_prototypes.shape, "centroids shape", reshaped_X.shape)
        # print("reshaped_numer_prototypes = " , reshaped_numer_prototypes, "shape = ", reshaped_numer_prototypes.shape)
        # print("reshaped_X = " , reshaped_X)
        difference = reshaped_numer_prototypes - reshaped_X                                                                               # Optional: use broadcasting to calculate the difference
        dist = np.sqrt(np.sum(np.power(difference, 2), axis = 2))                                                                                    # Required

        return dist

    def hamming_distance(self, X, is_categ, debug_prototypes=None):
        """
        Calculate the Hamming distance between each data point and prototypes. Only consider the categorical feature
        columns of each data point and prototypes.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param is_categ: ndarray of shape (n_features,), required
        Array of boolean values indicates whether each feature column is categorical or not.
        e.g., is_categ=[False, False, True, True] indicates that feature 2 and feature 3 of dataset X are categorical.
        Then, the remaining feature columns (i.e., feature 0 and feature 1) are numerical.
        :param debug_prototypes: ndarray of shape (k, n_features), optional
        If debug_prototypes is given, it will be used in the calculation rather than the stored prototypes in the Class
        Object. This argument is only used to help you test the function independently. Do not use this argument in the
        fit_predict function.

        :return: ndarray of shape (n_prototypes, n_samples)
        The Hamming distance between each data point and prototypes.
        """
        prototypes = debug_prototypes if debug_prototypes is not None else self.prototypes
        # TODO: only use the categorical feature columns to calculate the Hamming distance between each data point and
        #       prototypes.
        # Hints:
        #   - Notice that X and prototypes have mismatched shapes.
        #     X: (n_samples, n_features)                    prototypes: (n_prototypes, n_features)
        #   - Only the categorical feature columns are used to calculate the Hamming distance.
        #     X: (n_samples, n_categorical_features)        prototypes: (n_prototypes, n_categorical_features)
        #   - You may use Numpy's 'count_nonzero', 'reshape', 'sum', and 'not_equal' functions.
        #   - Try broadcasting.
        n_categ_features = np.sum(is_categ == True)                                                                         # Optional: number of categorical feature columns
        # print("prototypes shape = ", prototypes.shape, "is_categ = ", is_categ.shape)
        reshaped_categ_prototypes = np.delete(prototypes, is_categ == False, axis = 1)                                                             # Optional: reshape the categorical prototypes so that we can do broadcasting later
        reshaped_categ_prototypes = reshaped_categ_prototypes[:, np.newaxis]
        reshaped_X =  np.delete(X, is_categ == False, axis = 1) 
        # print("reshaped_categ_prototypes = " , reshaped_categ_prototypes)
        # print("reshaped_X = " , reshaped_X)
        difference =  np.not_equal(reshaped_X, reshaped_categ_prototypes)                                                                              # Optional: use broadcasting to calculate the difference
        dist = np.sum(difference, axis = 2)                                                                                    # Required

        return dist

    def fit_predict(self, X, is_categ):
        '''
        Compute cluster centers and predict cluster index for each sample.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param is_categ: ndarray of shape (n_features,), required
        Array of boolean values indicates whether each feature column is categorical or not.
        e.g., is_categ=[False, False, True, True] indicates that feature 2 and feature 3 of dataset X are categorical.
        Then, the remaining feature columns (i.e., feature 0 and feature 1) are numerical.

        :return: ndarray of shape (n_samples,)
        Index of the cluster (serve as the label) each sample belongs to.
        '''
        prev_prototypes = None
        iteration = 0
        # print("selected_prototypes = ", self.prototypes)
        # TODO: Set the criterion to leave the loop.
        # Hints:
        #   - The criterion to leave the loop is to satisfy either of the two conditions:
        #     1. Convergence criterion: the prototypes are the same as those in the last iteration.
        #     2. Max number of iterations: the algorithm runs to the max number of iterations, i.e., self.max_iter
        #   - You may use Numpy's 'not_equal' and 'any' function.
        while np.any(np.not_equal(prev_prototypes, self.prototypes)) and iteration < self.max_iter: # <= or <????
            # TODO: Assign the index of the closest prototype to each data point.
            # Hints: You may use numpy.argmin function to find the index of the closest prototype for each data point.
            numer_dist = self.euclidean_distance(X, is_categ).T
            # print("numer_dist = ", numer_dist)
            categ_dist = self.hamming_distance(X, is_categ).T
            # print("categ_dist = ", categ_dist)
            dist = numer_dist + categ_dist
            # print("dist = ", dist)
            prototype_idx = np.argmin(dist, axis=1)
            # print("prototype_idx", prototype_idx)
            prev_prototypes = self.prototypes.copy()  # Push current prototypes to previous.

            # TODO: Reassign prototypes as the mean of the clusters.
            # Hints:
            #  - We mentioned a method to choose specific elements from an array.
            #  - We mentioned that there were lots of functions from NumPy or scipy.stats for statistics.
            #    mean, std, median, mode, etc. On what axis should we find the statistics?
            #  - 'np.mean' and 'stats.mode' has different return shape. See how 'np.squeeze' works.
            for i in range(self.k):
                # A boolean array of shape (n_samples,). e.g., [False, True] means the second data sample is assigned to
                # cluster i but the first data sample is not.
                # print("start iter")
                assigned_idx = (prototype_idx == i)
                # print("assigned_idx", assigned_idx)
                if np.count_nonzero(assigned_idx) == 0:
                    continue

                if np.count_nonzero(~is_categ) > 0: # means that have to update the euclidian center
                    # Update the prototypes
                    # print("X.shape = ", X.shape, "assigned_idx.shape = ", assigned_idx.shape, "is_categ.shape = ", is_categ.shape)
                    new_X = np.delete(X, is_categ == True, axis = 1)
                    # print("new X = ", new_X)
                    new_X = np.delete(new_X, assigned_idx == False, axis = 0)
                    # print("new X = ", new_X)
                    mean_temp = np.mean(new_X, axis = 0)
                    # print("mean_temp ", mean_temp)
                    # print("self_prototype", self.prototypes)
                    self.prototypes[i][is_categ == False] = mean_temp
                    # print("self_prototype", self.prototypes)
                    # print("done iter")

                if np.count_nonzero(is_categ) > 0:  # means that have to update the categorical center
                    # The returned ndarray of stats.mode does not have the same shape as the 'np.mean' function.
                    # Please set the keepdims parameter of stats.mode function to avoid Warning. 
                    # print("X.shape = ", X.shape, "assigned_idx.shape = ", assigned_idx.shape, "is_categ.shape = ", is_categ.shape)
                    new_X = np.delete(X, is_categ == False, axis = 1)
                    # print("new X = ", new_X)
                    new_X = np.delete(new_X, assigned_idx == False, axis = 0)
                    # print("new X = ", new_X)
                    mode_temp, _ = stats.mode(new_X, keepdims = False) # debug???
                    # print("mode_temp ", mode_temp)
                    # print("self_prototype", self.prototypes)
                    # Convert this returned ndarray to the same shape as the 'np.mean' function before updating the prototypes.
                    self.prototypes[i][is_categ == True] = mode_temp
                    # print("self_prototype", self.prototypes)
                    # print("done iter")

            iteration += 1
        # print("------------------------------------------------")
        return prototype_idx


### Task 2: Evaluation Metrics, Sum of Squared Errors
def SSE(X, y, k, centroids):
    sse = 0
    # TODO: For each cluster, calculate the distance (square of difference, i.e. Euclidean/L2-distance) of samples to
    #  the datapoints and accumulate the sum to `sse`. (Hints: use numpy.sum and for loop)
    # Hints:
    #   - X is a Numpy 2D array with shape (num_datapoints, ndim), representing the data points.
    #   - y is a Numpy 1D array with shape (num_datapoints, ), representing which cluster (or which centroid) each data
    #   point correspond to.
    #   - This is very similar to the distance functions in Task 1
    sse = 0
    for i in range(k):
        assigned_idx = (y == i)
        # print("assigned_idx = ", assigned_idx)
        new_X = np.delete(X, assigned_idx == False, axis = 0)
        # print("new_X", new_X) #trimmed x
        reshaped_X =  new_X[:, np.newaxis] 
        # print("reshaped_X = ", reshaped_X)
        # print("centroids = ", centroids[i])
        # print("reshaped_X shape= ", reshaped_X.shape, "centroids shape", centroids.shape)
        difference = reshaped_X - centroids[i]                                                                   # Optional: use broadcasting to calculate the difference
        # print("difference = ",difference)
        sse += np.sum(np.power(difference, 2))
    return sse


