# Kernel ridge regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

from PythonMlTools.Tqdm import get_tqdm
tqdm = get_tqdm()

def FuncRandomForestRegressorKfold( n_estimators_list,
                                    max_depth_list,
                                    X_train, 
                                    y_train,
                                    K=3,
                                    random_state=0,
                                    min_samples_split=2,
                                    min_samples_leaf=2):
    found=False; k=0; 
    pbar=tqdm(range(np.size(n_estimators_list)));
    SCORE_P1P2=np.zeros((np.size(n_estimators_list),np.size(max_depth_list)))
    for j in pbar:
        n_estimators=n_estimators_list[j]
        score_val=[];
        ng=0;
        for max_depth in max_depth_list:
            rfr = RandomForestRegressor(    n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=random_state,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf);
            cv = KFold(n_splits=K, random_state=random_state, shuffle=True);
            #rfr.fit(X_train, y_train);
            
            scores = cross_val_score(rfr, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1)
            
            #st=rfr.score(X_train, y_train);
            sv=np.mean(scores);
            sv_std=np.std(scores);
            SCORE_P1P2[j][ng]=sv;
            
            score_val.append(sv);
            if k==0:
                n_estimators_opt=n_estimators;
                max_depth_opt=max_depth;
                score_opt=sv;
                rfr_opt=rfr;
                
                found=True;
            else:
                if sv>score_opt:
                    n_estimators_opt=n_estimators;
                    max_depth_opt=max_depth;
                    score_opt=sv;
                    rfr_opt=rfr;
                    
                    cad="";
                    cad=cad+"R^2 val: %.3f" % sv;
                    cad=cad+" (%.3f)" % sv_std;
                    cad=cad+"\tn_estimators:%d" % n_estimators;
                    cad=cad+"\tmax_depth:%d" % max_depth;
                    pbar.set_description(cad);
                    #pbar.set_description("R^2 val:"+str(sv)+" ("+str(sv_std)+")\tn_estimators:"+str(n_estimators)+"\tmax_depth:"+str(max_depth));
                    found=True;
            k=k+1
            ng=ng+1;
        if(found):
            score_val_opt=score_val.copy();
            found=False
    
    rfr_opt.fit(X_train, y_train);
    print("\nR^2 train+val:",rfr_opt.score(X_train, y_train));
    
    #print("rfr_opt:\n",rfr_opt.get_params(),"\n")
    
    return rfr_opt, n_estimators_opt, max_depth_opt, score_val_opt, SCORE_P1P2

import matplotlib.pyplot as plot
from scipy.stats import pearsonr

def FuncPlotDataKfold(  rfr_opt, 
                        n_estimators_opt, 
                        max_depth_opt, 
                        score_val_opt,
                        mean_y,std_y,
                        max_depth_list,
                        X_test, y_test,
                        Line=True):
    plot.figure(figsize=(6, 5))
    plot.plot(max_depth_list, score_val_opt);
    plot.xlabel('max_depth.');
    plot.ylabel('R^2 val');
    plot.title("n_estimators:"+str(n_estimators_opt))

    print(" ")
    print("n_estimators_opt: ",n_estimators_opt,"\tmax_depth_opt:",max_depth_opt);
    print(" ")
    
    R2_val=np.max(score_val_opt);
    R2_test=rfr_opt.score(X_test, y_test);
    print("R^2 is the coefficient of determination of the prediction. <-infty,1.0]")
    
    print("R^2 val  :", R2_val)
    print("R^2 test :", R2_test)
    
    Ypred=std_y*rfr_opt.predict(X_test)+mean_y;
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
    
def FuncContourFRfrDataKfold(  n_estimators_list,
                            max_depth_list,
                            SCORE_P1P2,
                            nlevels=32,
                            title='',
                            cmap_str='jet'):
    # plot
    fig, ax = plot.subplots(figsize=(12, 12));
    XXX, YYY = np.meshgrid(n_estimators_list, max_depth_list)
    #print('SCORE_P1P2.shape:',SCORE_P1P2.shape);
    #print('     XXX.shape:',XXX.shape);
    #print('     YYY.shape:',YYY.shape);
    #im=ax.pcolormesh(XXX.T, YYY.T, SCORE_P1P2);
    levels = np.linspace(SCORE_P1P2.min(), SCORE_P1P2.max(), nlevels);
    im=ax.contourf(XXX.T, YYY.T, SCORE_P1P2,levels=levels,cmap=cmap_str);
    ax.set_xlabel('n_estimators');
    ax.set_ylabel('max_depth');
    ax.set_title(title);
    plot.colorbar(im,label="R2", orientation="vertical") ;
    plot.show();

def FuncSurfaceRfrDataKfold(n_estimators_list,
                            max_depth_list,
                            SCORE_P1P2,
                            title='',
                            cmap_str='jet'):
    XXX, YYY = np.meshgrid(n_estimators_list, max_depth_list);
    # plot
    fig = plot.figure(figsize=(12, 12));
    ax = plot.axes(projection='3d');
    ax.plot_surface(XXX.T, YYY.T, 
                    SCORE_P1P2, 
                    rstride=1, 
                    cstride=1, 
                    cmap=cmap_str#, edgecolor='none'
                   );
    ax.set_xlabel('n_estimators');
    ax.set_ylabel('max_depth');
    ax.set_zlabel('R2');
    ax.set_title(title);
    #ax.view_init(60, 35);
    plot.show();
