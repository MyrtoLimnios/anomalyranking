"""
 Script for generating datasets for main_aanomalyranking of the paper
   " Learning to Rank Anomalies: Scalar Performance Criteria and Maximization of Rank Statistics "

 author: Myrto Limnios // mail: myli@math.ku.dk

"""
import scipy.stats as stats
import numpy as np
import random as rd

import random
rds = 4
random.seed(rds)


def XY_generator(n, m, d, eps, sample_type, distrib):
    if distrib == 'Gaussian':

        if sample_type == 'G1':
            # mu_X =  0.6 * np.ones(d)
            mu_X = np.zeros(d)
            b = eps
            S_Y = 0.05 * np.array([[b ** np.abs(i - j) for i in range(d)] for j in range(d)])
            S_X = S_Y


        if sample_type == 'G2-':
            mu_X = eps*np.ones(d)
            if d == 4:
                S_X = 0.01*np.array([[2, -1, -1, -1], [-1, 6, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 5]])
            if d == 6:
                 S_X = 0.01*np.array([[2, -1, -1, -1, -1, -1], [-1, 6, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0], [-1, 0, 0, 5, 0, 0],
                       [-1, 0, 0, 0, 4, 0], [-1, 0, 0, 0, 0, 3]])


        if sample_type == 'G2+':
            mu_X = eps*np.ones(d)
            if d == 4:
                S_X = 0.01*np.array([[6, -2, -3, -2], [-2, 5, 4, 0],[-3, 4, 5, 2], [-2, 0, 2, 8]])
            if d == 6:
                S_X = 0.01*np.array([[10, -.2, -1, -.2, -1, -2 ], [-.2, 8, 1, 0, -1, 2],[-1, 1, 10, 2, 0, 1], [-.2, 0, 2, 8, 1, 1],
                          [-1, -1, 0, 1, 12, 0], [-2, 2, 1, 1, 0, 15]])
            S_Y = S_X

        if sample_type == 'G3':
            b = 0.3
            a = b + eps
            mu_X = np.zeros(d)
            S_X = (1 - a) * np.eye(d) + a * np.ones((d, d))
            S_X = 0.01*S_X


    X = np.random.multivariate_normal(mu_X, S_X, n)
    Y = np.random.uniform(low=-1, high=1, size=(m, d))
    XY = np.concatenate((X, Y)).astype(np.float32)

    scor = np.concatenate((np.ones(n), np.zeros(m))).astype(np.int32)

    return XY, scor, mu_X, S_X


