"""
 Main script for learning to rank anomalies
 Use datagenerator_MV to generate the samples with the probabilistic models of the paper
   " Learning to Rank Anomalies: Scalar Performance Criteria and Maximization of Rank Statistics "


 author: Myrto Limnios // mail: myli@math.ku.dk

 What it does: based on two simulated train samples, outputs the optimal scored test sample by learning a bipartite ranking model.
 Also LogisticRegression and Local Outlier Factor are implemented as state-of-the-art methods. For set of parameters, it creates a csv file were the
 optimal scores for the test sample is stored at every Monte-Carlo loop.

"""

import treerank as TR

import numpy as n
import datagenerator_MV as data

import scipy.stats as stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression

from RankBoost import RankB

import datetime
import random
rds = 4
random.seed(rds)

'''  generate two-sample data '''
distrib = 'Gaussian'
sample_type = ''


eps_range = []
param_range = eps_range

'''   Complete with the path to your directory '''
path = r''

B = 20  # MC samplings

for m in []:
    n =
    ntst, mtst = ,

    df_ = {}
    for d in []:
        """ Name the file """
        print('loop', 'sample size', m, 'd', d)
        str_param = sample_type + str(n) + str(m) + str(d) + str(int(ntst)) + str(int(mtst)) + str(B)

        '''  Loop on eps and save the results in a csv file '''
        for eps in np.around(eps_range, decimals=2):

            names_ = []
            names = []
            for nam in names_:
                for ktree in range(K_tree):
                    names.append(nam + '_' + str(ktree))

            s_predrk_list = []
            y_pred = []
            y_pred_rlr = []
            y_pred_knn_lab = []
            y_pred_knn_scor = []
            y_pred_rboost_scor = []

            noise = 0.9
            str_param += 'mtrain0' + str(int(eps * 10)) + 'mtest0' + str(int(eps * 10)) + 'dpth10' + \
                         'gauss' + 'eps' + str(int(noise * 100)) + 'neg' + 'std01' #+ 'u03'

            for b in range(B):
                '''  Generate the data: XY matrix of the twosamples, with scor=1 for X and scor=0 for Y, q=unit vector '''
                print('Generate datasets')

                x_train, scor_train, mu_X, S_X = data.XY_generator(n, m, d, eps, sample_type, distrib)

                x_test, scor_test, _, _ = data.XY_generator(int(ntst * noise), mtst, d, eps, sample_type, distrib)

                """ If Gaussian Noise"""
                #e = np.random.multivariate_normal((eps - 0.5) * np.ones(d), 0.01 * np.diag(np.ones(d)), ntst - len(x_test))
                """ If Uniform Noise"""
                #e = np.random.uniform(low=-1., high=1., size=(ntst - len(x_test), d))

                x_test = np.concatenate((x_test, e)).astype(np.float32)
                scor_test = np.ones(ntst)

                """ TreeRank"""
                print("#"*80, "#{:^78}#".format("TREE"), "#"*80, sep='\n')
                tree = TR.TreeRANK(max_depth=5, verbose=0)
                rank_list = tree.fit(x_train, scor_train)

                # Evaluate Testing
                y_pred.append(tree.predict_scor(x_test))

                """ RankBoost """
                rboost = RankB()
                rboost.fit(x_train, scor_train)
                predp = rboost.predict_proba(x_test)[:, 1]
                y_pred_rboost_scor.append(predp)

                """ Logistic Reegression """
                rLR = LogisticRegression(penalty='l2', tol=1e-6, C=100.)
                rLR.fit(x_train, scor_train)
                pred = rLR.predict_proba(x_test)[:, 1]
                y_pred_rlr.append(pred)

                """ LOF """
                nbrs = LocalOutlierFactor(n_neighbors=6, novelty=True).fit(x_train[0:n])
                y_pred_knn_scor.append(nbrs.score_samples(x_test[0:ntst] ))

            s_predrk_list = np.concatenate((y_pred, y_pred_rboost_scor, y_pred_rlr, y_pred_knn_scor))

            print(s_predrk_list)

            df_ = pd.DataFrame.from_dict(dict(zip(names, s_predrk_list)))
            df_.to_csv(path + 'scor_MV_all_' + datetime.datetime.today().strftime("%m%d%H%M") + str_param + '.csv')
            df_ = {}
