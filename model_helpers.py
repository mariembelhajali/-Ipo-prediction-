#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import warnings
import numpy as np
warnings.simplefilter('ignore')

# Extract feature importances
def feature_importance(pipeline,features,target):
    
    print('most important features: ')
    clf = pipeline.named_steps['gb_clf']
    clf.fit(features, target)
    i = 0
    for index in reversed(np.argsort(clf.feature_importances_)):
        print(features.columns[index] , ':', clf.feature_importances_[index])
        i = i + 1
        if (i == 15):
            break

def nested_cv(X, y, est_pipe, p_grid, p_score, n_splits_inner = 3, n_splits_outer = 3, n_cores = 1, seed = 0):

    # Cross-validation schema for inner and outer loops
    inner_cv = StratifiedKFold(n_splits = n_splits_inner, shuffle = True, random_state = seed)
    outer_cv = StratifiedKFold(n_splits = n_splits_outer, shuffle = True, random_state = seed)
    
    # Grid search to tune hyper parameters
    est = GridSearchCV(estimator = est_pipe, param_grid = p_grid, cv = inner_cv, scoring = p_score, n_jobs = n_cores)

    # Nested CV with parameter optimization
    nested_scores = cross_val_score(estimator = est, X = X, y = y, cv = outer_cv, scoring = p_score, n_jobs = n_cores)
    
    print('Average score: %0.4f (+/- %0.4f)' % (nested_scores.mean(), nested_scores.std() * 1.96))
    
warnings.filterwarnings('ignore')
seed = 0



## function to plot roc curve
def plot_roc_curve(model,y_pred,y_test):

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred[:, 1])

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label=model)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

