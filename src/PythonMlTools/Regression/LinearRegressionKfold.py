# Kernel ridge regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from tqdm.notebook import tqdm
import numpy as np

def FuncLinearRegressionKfoldBest(X_train, y_train,K=3):
    
    lr = LinearRegression();
    cv = KFold(n_splits=K, random_state=1, shuffle=True);
    #lr.fit(X_train, y_train);
    
    scores = cross_val_score(lr, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
    
    #st=lr.score(X_train, y_train);
    sv=np.mean(scores);
    sv_std=np.std(scores);
    
    lr.fit(X_train, y_train);
    print("\nR^2 train+val:",lr.score(X_train, y_train));
    
    return lr, sv, sv_std

