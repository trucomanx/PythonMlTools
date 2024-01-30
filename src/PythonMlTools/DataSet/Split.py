#!/usr/bin/python

#import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split_stratify_index(y,test_size=0.38, random_state=42):
    if y.ndim>1:
        sys.exit("Error in: train_test_split_stratify_index !!!!!!");
    y=np.round(y);
    N=np.size(y);
    X=np.array(range(0,N)).reshape((-1,1));
    X_train, X_test, y_train, y_test = train_test_split( X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state);
    y_train_id = X_train.reshape((-1,)); 
    y_test_id  = X_test.reshape((-1,));
    
    return y_train_id, y_test_id, y_train, y_test;
