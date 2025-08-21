# Kernel ridge regression
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from PythonMlTools.Tqdm import get_tqdm
tqdm = get_tqdm()

def FuncKernelRidgeBestGaussian(alpha_list,gamma_list,X_train, y_train,X_val, y_val):
    found=False; k=0; 
    pbar=tqdm(range(np.size(alpha_list)));
    for j in pbar:
        alpha=alpha_list[j]
        score_val=[];
        for gamma in gamma_list:
            krr = KernelRidge(alpha=alpha,kernel="rbf",gamma=gamma);
            krr.fit(X_train, y_train);
            st=krr.score(X_train, y_train);
            sv=krr.score(X_val, y_val);

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
                    
                    cad="";
                    cad=cad+"R^2 val: %.3f" % sv;
                    cad=cad+"\talpha:%.3e" % alpha;
                    cad=cad+"\tgamma:%.3e" % gamma;
                    pbar.set_description(cad);
                    #pbar.set_description("R^2 val:"+str(sv)+" \talpha:"+str(alpha)+"\tgamma:"+str(gamma));
                    found=True;
            k=k+1
        if(found):
            score_val_opt=score_val.copy();
            found=False
    
    print(" ");
    
    return krr_opt, alpha_opt, gamma_opt, score_val_opt

import matplotlib.pyplot as plot

def FuncPlotData(   krr_opt, 
                    alpha_opt, 
                    gamma_opt, 
                    score_val_opt,
                    mean_y,std_y,
                    gamma_list,
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test):
    plot.figure(figsize=(6, 5))
    plot.plot(gamma_list, score_val_opt);
    plot.xlabel('gamma.');
    plot.ylabel('R^2 val');
    plot.title("alpha:"+str(alpha_opt))

    print(" ")
    print("alpha_opt: ",alpha_opt,"\tgamma_opt:",gamma_opt);
    print(" ")
    R2_train=krr_opt.score(X_train, y_train);
    R2_val=krr_opt.score(X_val, y_val);
    R2_test=krr_opt.score(X_test, y_test);
    print("R^2 is the coefficient of determination of the prediction. <-infty,1.0]")
    print("R^2 train:", R2_train)
    print("R^2 val  :", R2_val)
    print("R^2 test :", R2_test)

    '''
    plot.figure(figsize=(6, 5));
    plot.scatter(std_y*y_train+mean_y,std_y*krr_opt.predict(X_train)+mean_y);
    plot.xlabel('train');
    plot.ylabel('predict');
    plot.title('R^2 (train): '+str(R2_train))
    
    plot.figure(figsize=(6, 5));
    plot.scatter(std_y*y_val+mean_y,std_y*krr_opt.predict(X_val)+mean_y);
    plot.xlabel('val');
    plot.ylabel('predict');
    plot.title('R^2 (val): '+str(R2_val))
    '''
    
    Ypred=std_y*krr_opt.predict(X_test)+mean_y;
    Yreal=std_y*y_test+mean_y;
    
    plot.figure(figsize=(6, 5));
    plot.scatter(Yreal,Ypred);
    plot.xlabel('test');
    plot.ylabel('predict');
    plot.title('R^2 (test): '+str(R2_test))
    
    plot.show()
    
    print("MAPE:\t",mean_absolute_percentage_error(Yreal,Ypred))
    
    return R2_train, R2_val, R2_test;
    
