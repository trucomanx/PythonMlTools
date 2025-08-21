# Kernel ridge regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

from PythonMlTools.Tqdm import get_tqdm
tqdm = get_tqdm()

def FuncSVCKfoldBestGaussian(epsilon_list,gamma_list,X_train, y_train,K=3):
    found=False; k=0; 
    pbar=tqdm(range(np.size(epsilon_list)));
    SCORE_EG=np.zeros((np.size(epsilon_list),np.size(gamma_list)))
    for j in pbar:
        epsilon=epsilon_list[j]
        score_val=[];
        ng=0;
        for gamma in gamma_list:
            ksvc = SVC(C=epsilon,kernel="rbf",gamma=gamma);
            cv = KFold(n_splits=K, random_state=1, shuffle=True);
            #ksvc.fit(X_train, y_train);
            
            scores = cross_val_score(ksvc, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
            
            #st=ksvc.score(X_train, y_train);
            sv=np.mean(scores);
            sv_std=np.std(scores);
            SCORE_EG[j][ng]=sv;
            
            score_val.append(sv);
            if k==0:
                epsilon_opt=epsilon;
                gamma_opt=gamma;
                score_opt=sv;
                ksvc_opt=ksvc;
                
                found=True;
            else:
                if sv>score_opt:
                    epsilon_opt=epsilon;
                    gamma_opt=gamma;
                    score_opt=sv;
                    ksvc_opt=ksvc;
                    
                    cad="";
                    cad=cad+"Acc. val: %.3f" % sv;
                    cad=cad+" (%.3f)" % sv_std;
                    cad=cad+"\tepsilon:%.3e" % epsilon;
                    cad=cad+"\tgamma:%.3e" % gamma;
                    pbar.set_description(cad);
                    
                    found=True;
            k=k+1
            ng=ng+1;
        if(found):
            score_val_opt=score_val.copy();
            found=False
    
    ksvc_opt.fit(X_train, y_train);
    print("\nAcc. train+val:",ksvc_opt.score(X_train, y_train));
    
    #print("ksvc_opt:\n",ksvc_opt.get_params(),"\n")
    
    return ksvc_opt, epsilon_opt, gamma_opt, score_val_opt, SCORE_EG

import matplotlib.pyplot as plot
from scipy.stats import pearsonr

def FuncPlotDataKfold(  ksvc_opt, 
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
    plot.ylabel('Acc. val');
    plot.title("epsilon:"+str(epsilon_opt))

    print(" ")
    print("epsilon_opt: ",epsilon_opt,"\tgamma_opt:",gamma_opt);
    print(" ")
    
    Acc_val=np.max(score_val_opt);
    Acc_test=ksvc_opt.score(X_test, y_test);
    print("Acc. is the coefficient of determination of the prediction. <-infty,1.0]")
    
    print("Acc. val  :", Acc_val)
    print("Acc. test :", Acc_test)
    
    Ypred=std_y*ksvc_opt.predict(X_test)+mean_y;
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
    plot.title('Acc. (test): '+str(Acc_test))
    
    plot.show()
    
    MAPE=mean_absolute_percentage_error(Yreal,Ypred);
    print("MAPE:\t",MAPE)
    corr, _ = pearsonr(Yreal,Ypred)
    print('Pearsons correlation: %.3f' % corr)
    
    return Acc_val, Acc_test, MAPE;
    
def FuncContourFSvcDataKfold(  epsilon_list,
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
    plot.colorbar(im,label="Acc", orientation="vertical") ;
    plot.show();

def FuncSurfaceSvcDataKfold(epsilon_list,
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
    ax.set_zlabel('Acc');
    ax.set_title(title);
    #ax.view_init(60, 35);
    plot.show();
