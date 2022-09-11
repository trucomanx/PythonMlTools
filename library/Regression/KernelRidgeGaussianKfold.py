# Kernel ridge regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm, trange
import numpy as np

def FuncKernelRidgeKfoldBestGaussian(alpha_list,gamma_list,X_train, y_train,K=3,verbose=True):
    found=False; k=0; 
    for j in trange(np.size(alpha_list)):
        alpha=alpha_list[j]
        score_val=[];
        for gamma in gamma_list:
            krr = KernelRidge(alpha=alpha,kernel="rbf",gamma=gamma);
            cv = KFold(n_splits=K, random_state=1, shuffle=True);
            #krr.fit(X_train, y_train);
            
            scores = cross_val_score(krr, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
            
            #st=krr.score(X_train, y_train);
            sv=np.mean(scores);
            sv_std=np.std(scores);
            
            score_val.append(sv);
            if k==0:
                alpha_opt=alpha;
                gamma_opt=gamma;
                score_opt=sv;
                krr_opt=krr;
                
                found=True;
            else:
                if sv>score_opt:
                    alpha_opt=alpha;
                    gamma_opt=gamma;
                    score_opt=sv;
                    krr_opt=krr;
                    if(verbose):
                        print("R^2 val:",sv," (",sv_std,")","\talpha:",alpha,"\tgamma:",gamma)
                    else:
                        print("R^2 val:",sv," (",sv_std,")","\talpha:",alpha,"\tgamma:",gamma,"\t\t",end="\r")
                    found=True;
            k=k+1
        if(found):
            score_val_opt=score_val.copy();
            found=False
    
    krr_opt.fit(X_train, y_train);
    print("\nR^2 train+val:",krr_opt.score(X_train, y_train));
    
    print("krr_opt:\n",krr_opt.get_params(),"\n")
    
    return krr_opt, alpha_opt, gamma_opt, score_val_opt

import matplotlib.pyplot as plot

def FuncPlotDataKfold(  krr_opt, 
                        alpha_opt, 
                        gamma_opt, 
                        score_val_opt,
                        mean_y,std_y,
                        gamma_list,
                        X_test, y_test):
    plot.figure(figsize=(6, 5))
    plot.plot(gamma_list, score_val_opt);
    plot.xlabel('gamma.');
    plot.ylabel('R^2 val');
    plot.title("alpha:"+str(alpha_opt))

    print(" ")
    print("alpha_opt: ",alpha_opt,"\tgamma_opt:",gamma_opt);
    print(" ")
    
    R2_val=np.max(score_val_opt);
    R2_test=krr_opt.score(X_test, y_test);
    print("R^2 is the coefficient of determination of the prediction. <-infty,1.0]")
    
    print("R^2 val  :", R2_val)
    print("R^2 test :", R2_test)
    
    Ypred=std_y*krr_opt.predict(X_test)+mean_y;
    Yreal=std_y*y_test+mean_y;
    
    plot.figure(figsize=(6, 5));
    plot.scatter(Yreal,Ypred);
    plot.xlabel('test');
    plot.ylabel('predict');
    plot.title('R^2 (test): '+str(R2_test))
    
    plot.show()
    
    print("MAPE:\t",mean_absolute_percentage_error(Yreal,Ypred))
    
    return R2_val, R2_test;
    
