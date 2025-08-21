# Kernel ridge regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

from PythonMlTools.Tqdm import get_tqdm
tqdm = get_tqdm()

def FuncSVRKfoldBestGaussian(epsilon_list,gamma_list,X_train, y_train,K=3):
    found=False; k=0; 
    pbar=tqdm(range(np.size(epsilon_list)));
    SCORE_EG=np.zeros((np.size(epsilon_list),np.size(gamma_list)))
    for j in pbar:
        epsilon=epsilon_list[j]
        score_val=[];
        ng=0;
        for gamma in gamma_list:
            krr = SVR(epsilon=epsilon,kernel="rbf",gamma=gamma);
            cv = KFold(n_splits=K, random_state=1, shuffle=True);
            #krr.fit(X_train, y_train);
            
            scores = cross_val_score(krr, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
            
            #st=krr.score(X_train, y_train);
            sv=np.mean(scores);
            sv_std=np.std(scores);
            SCORE_EG[j][ng]=sv;
            
            score_val.append(sv);
            if k==0:
                epsilon_opt=epsilon;
                gamma_opt=gamma;
                score_opt=sv;
                krr_opt=krr;
                
                found=True;
            else:
                if sv>score_opt:
                    epsilon_opt=epsilon;
                    gamma_opt=gamma;
                    score_opt=sv;
                    krr_opt=krr;
                    
                    cad="";
                    cad=cad+"R^2 val: %.3f" % sv;
                    cad=cad+" (%.3f)" % sv_std;
                    cad=cad+"\tepsilon:%.3e" % epsilon;
                    cad=cad+"\tgamma:%.3e" % gamma;
                    pbar.set_description(cad);
                    #pbar.set_description("R^2 val:"+str(sv)+" ("+str(sv_std)+")\tepsilon:"+str(epsilon)+"\tgamma:"+str(gamma));
                    found=True;
            k=k+1
            ng=ng+1;
        if(found):
            score_val_opt=score_val.copy();
            found=False
    
    krr_opt.fit(X_train, y_train);
    print("\nR^2 train+val:",krr_opt.score(X_train, y_train));
    
    #print("krr_opt:\n",krr_opt.get_params(),"\n")
    
    return krr_opt, epsilon_opt, gamma_opt, score_val_opt, SCORE_EG

import matplotlib.pyplot as plot
from scipy.stats import pearsonr

def FuncPlotDataKfold(  krr_opt, 
                        epsilon_opt, 
                        gamma_opt, 
                        score_val_opt,
                        mean_y,std_y,
                        gamma_list,
                        X_test, y_test,
                        Line=True):
    plot.figure(figsize=(6, 5))
    plot.plot(gamma_list, score_val_opt);
    plot.xlabel('gamma.');
    plot.ylabel('R^2 val');
    plot.title("epsilon:"+str(epsilon_opt))

    print(" ")
    print("epsilon_opt: ",epsilon_opt,"\tgamma_opt:",gamma_opt);
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
    if Line:
        lr_tt = LinearRegression();
        lr_tt.fit(Yreal.reshape(-1,1),Ypred);
        Yfake=lr_tt.predict(Yreal.reshape(-1,1));
        plot.plot(Yreal,Yfake,label='{0:.3f}'.format(lr_tt.coef_[0])+'Yreal+'+'{0:.3f}'.format(lr_tt.intercept_) );
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
    corr, _ = pearsonr(Yreal,Ypred)
    print('Pearsons correlation: %.3f' % corr)
    
    return R2_val, R2_test, MAPE;
    
def FuncContourFSvrDataKfold(  epsilon_list,
                            gamma_list,
                            SCORE_EG,
                            nlevels=32,
                            title='',
                            cmap_str='jet'):
    # plot
    fig, ax = plot.subplots(figsize=(12, 12));
    XXX, YYY = np.meshgrid(epsilon_list, gamma_list)
    #print('SCORE_EG.shape:',SCORE_EG.shape);
    #print('     XXX.shape:',XXX.shape);
    #print('     YYY.shape:',YYY.shape);
    #im=ax.pcolormesh(XXX.T, YYY.T, SCORE_EG);
    levels = np.linspace(SCORE_EG.min(), SCORE_EG.max(), nlevels);
    im=ax.contourf(XXX.T, YYY.T, SCORE_EG,levels=levels,cmap=cmap_str);
    ax.set_xlabel('epsilon');
    ax.set_ylabel('gamma');
    ax.set_title(title);
    plot.colorbar(im,label="R2", orientation="vertical") ;
    plot.show();

def FuncSurfaceSvrDataKfold(epsilon_list,
                            gamma_list,
                            SCORE_EG,
                            title='',
                            cmap_str='jet'):
    XXX, YYY = np.meshgrid(epsilon_list, gamma_list);
    # plot
    fig = plot.figure(figsize=(12, 12));
    ax = plot.axes(projection='3d');
    ax.plot_surface(XXX.T, YYY.T, 
                    SCORE_EG, 
                    rstride=1, 
                    cstride=1, 
                    cmap=cmap_str#, edgecolor='none'
                   );
    ax.set_xlabel('epsilon');
    ax.set_ylabel('gamma');
    ax.set_zlabel('R2');
    ax.set_title(title);
    #ax.view_init(60, 35);
    plot.show();
