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


import matplotlib.pyplot as plot

def FuncPlotDataKfold(  lr_opt, 
                        R2_val,
                        mean_y,std_y,
                        X_test, y_test):
    plot.figure(figsize=(6, 5))
    
    R2_test=lr_opt.score(X_test, y_test);
    print("R^2 is the coefficient of determination of the prediction. <-infty,1.0]")
    
    print("R^2 val  :", R2_val)
    print("R^2 test :", R2_test)
    
    Ypred=std_y*lr_opt.predict(X_test)+mean_y;
    Yreal=std_y*y_test+mean_y;
    
    lr_tt = LinearRegression();
    lr_tt.fit(Yreal.reshape(-1,1),Ypred);
    Yfake=lr_tt.predict(Yreal.reshape(-1,1));
    
    plot.figure(figsize=(6, 5));
    plot.scatter(Yreal,Ypred,label='(real,predict)');
    plot.plot(Yreal,Yfake,label=np.array2string(lr_tt.coef_)+'Ypred+'+str(lr_tt.intercept_));
    plot.legend()
    
    MIN=np.min([Ypred.min(),Yreal.min()]); 
    MAX=np.max([Ypred.max(),Yreal.max()]);
    plot.xlim(MIN,MAX);
    plot.ylim(MIN,MAX);
    plot.xlabel('real');
    plot.ylabel('predict');
    plot.title('R^2 (test): '+str(R2_test))
    
    plot.show()
    
    MAPE=mean_absolute_percentage_error(Yreal,Ypred);
    print("MAPE:\t",MAPE)
    
    return R2_val, R2_test, MAPE;
    
