#!/usr/bin/python
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_vec_xy(  VEC_XY,
                      labels_x=None,
                      label_y='',
                      title='',
                      figxsize=15,
                      figysize=4,
                      img_filepath=None,
                      horizontal=True):
    
    if horizontal:
        fig = plt.figure(figsize=(figxsize, figysize)) # width and height in inches
        plt.bar(labels_x, VEC_XY, edgecolor="white", linewidth=0.7,tick_label=labels_x)
        plt.xticks(rotation = 90)
    else:
        fig = plt.figure(figsize=(figysize, figxsize)) # width and height in inches
        plt.barh(labels_x, VEC_XY, edgecolor="white", linewidth=0.7)
    if isinstance(label_y,str):
        plt.ylabel(label_y);
    plt.title(title);
    
    try:
        fig.savefig(img_filepath);
    except:
        pass;
    plt.show();

def plot_mat_xy(  MAT_XY,
                  labels_x=None,
                  labels_y=None,
                  title='',
                  figxsize=15,
                  figysize=2,
                  img_filepath=None,
                  cmap='jet'):

    if len(MAT_XY.shape)!=2:
        sys.exit('Problem with shapes: len(MAT_XY.shape)!=2 : current length = '+str(len(MAT_XY.shape)))

    fig = plt.figure(figsize=(figysize, figxsize)) # width and height in inches
    if labels_x==None and labels_y==None:
        sns.heatmap(MAT_XY,
                    cmap=cmap)
    elif labels_x==None and labels_y!=None:
        sns.heatmap(MAT_XY,
                    xticklabels=labels_y, 
                    cmap=cmap)
    elif labels_x!=None and labels_y==None:
        sns.heatmap(MAT_XY,
                    yticklabels=labels_x,
                    cmap=cmap)
    else:
        sns.heatmap(MAT_XY,
                    yticklabels=labels_x, 
                    xticklabels=labels_y,
                    cmap=cmap)
    plt.title(title)
    try:
        fig.savefig(img_filepath);
    except:
        pass;
    plt.show();

################################################################################
## Correlation analysis
################################################################################

def plot_corrcoef_x(X_in,
                    labels_x=None,
                    title='',
                    figsize=15,
                    img_filepath=None,
                    cmap='jet'):
    if len(X_in.shape)==1:
        Nx=1;
    else:
        Nx=X_in.shape[1];
    
    X=X_in.reshape((-1,Nx));
    
    MAT_X=np.corrcoef(X.T);
    
    plot_mat_xy(  MAT_X,
                  labels_x=labels_x,
                  labels_y=labels_x,
                  title=title,
                  figxsize=figsize,
                  figysize=figsize,
                  img_filepath=img_filepath,
                  cmap=cmap);
    return MAT_X;

def plot_corrcoef_xy(   X_in,
                        Y_in,
                        labels_x=None,
                        labels_y=None,
                        title='',
                        figxsize=15,
                        figysize=2,
                        img_filepath=None,
                        cmap='jet',
                        horizontal=False):
    if len(X_in.shape)==1:
        Nx=1;
    else:
        Nx=X_in.shape[1];
    
    if len(Y_in.shape)==1:
        Ny=1;
    else:
        Ny=Y_in.shape[1];
    X=X_in.reshape((-1,Nx));
    Y=Y_in.reshape((-1,Ny));

    if X.shape[0]!=Y.shape[0]:
        sys.exit('Problem with shapes: X_in.shape[0]!=Y_in.shape[0]')
    L =X.shape[0];

    CORR_XY=np.zeros((Nx,Ny))

    for m in range(Nx):
        for n in range(Ny):
            corr=np.abs(np.corrcoef(X[:,m],Y[:,n]))[0,1]
            CORR_XY[m,n]=corr;
    
    if horizontal:
        plot_mat_xy(  CORR_XY.T,
                      labels_x=labels_y,
                      labels_y=labels_x,
                      title=title,
                      figxsize=figysize,
                      figysize=figxsize,
                      img_filepath=img_filepath,
                      cmap=cmap);
    else:
        plot_mat_xy(  CORR_XY,
                      labels_x=labels_x,
                      labels_y=labels_y,
                      title=title,
                      figxsize=figxsize,
                      figysize=figysize,
                      img_filepath=img_filepath,
                      cmap=cmap);
    return CORR_XY;
    

################################################################################
## Mutual information
################################################################################
import PythonMlTools.FeatureSelection.Analysis.Information as AnI

def plot_mutual_info_bins_x(    X_in,
                                bins,
                                bandwidth,#=0.8/bins,
                                labels_x=None,
                                title='',
                                figsize=15,
                                img_filepath=None,
                                cmap='jet'):
    if len(X_in.shape)==1:
        Nx=1;
    else:
        Nx=X_in.shape[1];
    
    X=X_in.reshape((-1,Nx));
    
    mat_x  = AnI.all_against_all_mutual_inf(X,bins=bins,bandwidth=bandwidth);
    plot_mat_xy(  mat_x,
                  labels_x=labels_x,
                  labels_y=labels_x,
                  title=title,
                  figxsize=figsize,
                  figysize=figsize,
                  img_filepath=img_filepath,
                  cmap=cmap)
    return mat_x 

def plot_sorted_mutual_info_bins_xy(    X_in,
                                        y_in,
                                        bins,
                                        bandwidth,#=0.8/bins,
                                        labels_x=None,
                                        title='',
                                        figxsize=3,
                                        figysize=15,
                                        img_filepath=None):
    if len(y_in.shape)!=1:
        sys.exit('Problem with len(y_in.shape)!=1');
    
    if len(X_in.shape)==1:
        Nx=1;
    else:
        Nx=X_in.shape[1];
    
    X=X_in.reshape((-1,Nx));
    
    mat_xy = AnI.x_against_y_mutual_inf(X,y_in.reshape((-1,1)),bins=bins,bandwidth=bandwidth);
    
    if labels_x==None:
        labels_x=['f'+str(n) for n in range(Nx)];
        
    Zdat = sorted(zip(mat_xy.tolist(),labels_x));#, reverse=True
    
    DATA_COLUMNS_SORTED_m = [a for _,a in Zdat];
    Data_SORTED           = [a for a,_ in Zdat];
    
    mat_xy_SORTED_m       = np.array(Data_SORTED).reshape((-1,));
    
    
    plot_bar_vec_xy(  mat_xy_SORTED_m,
                      labels_x=DATA_COLUMNS_SORTED_m,
                      label_y='',
                      title=title,
                      figxsize=figxsize,
                      figysize=figysize,
                      img_filepath=img_filepath);
    
    return mat_xy_SORTED_m;

#############################################################
from sklearn.feature_selection import mutual_info_regression

def plot_mutual_info_regression_x(  X_in,
                                    labels_x=None,
                                    title='',
                                    figsize=15,
                                    img_filepath=None,
                                    cmap='jet'):
    if len(X_in.shape)==1:
        Nx=1;
    else:
        Nx=X_in.shape[1];
    
    X=X_in.reshape((-1,Nx));
    
    MI_X=np.zeros((Nx,Nx))

    for n in range(Nx):
        for m in range(Nx):
            MI_X[n,m]=mutual_info_regression(X[:,n].reshape((-1,1)), X[:,m])[0];
    
    MI_X=MI_X/np.max(MI_X);
    
    plot_mat_xy(  MI_X.T,
                  labels_x=labels_x,
                  labels_y=labels_x,
                  title=title,
                  figxsize=figsize,
                  figysize=figsize,
                  img_filepath=img_filepath,
                  cmap=cmap);
    
    return MI_X;
    
def plot_mutual_info_regression_xy( X_in,
                                    Y_in,
                                    labels_x=None,
                                    labels_y=None,
                                    title='',
                                    figxsize=15,
                                    figysize=2,
                                    img_filepath=None,
                                    cmap='jet',
                                    horizontal=False):
    if len(X_in.shape)==1:
        Nx=1;
    else:
        Nx=X_in.shape[1];
    
    if len(Y_in.shape)==1:
        Ny=1;
    else:
        Ny=Y_in.shape[1];
    X=X_in.reshape((-1,Nx));
    Y=Y_in.reshape((-1,Ny));

    if X.shape[0]!=Y.shape[0]:
        sys.exit('Problem with shapes: X_in.shape[0]!=Y_in.shape[0]')
    L =X.shape[0];

    MI_XY=np.zeros((Nx,Ny))
    
    MAX=mutual_info_regression(Y[:,0].reshape((-1,1)), Y[:,0]);
    for n in range(Ny):
        MI_XY[:,n]=mutual_info_regression(X, Y[:,n])/MAX;
    
    if horizontal:
        plot_mat_xy(  MI_XY.T,
                      labels_x=labels_y,
                      labels_y=labels_x,
                      title=title,
                      figxsize=figysize,
                      figysize=figxsize,
                      img_filepath=img_filepath,
                      cmap=cmap);
    else:
        plot_mat_xy(  MI_XY,
                      labels_x=labels_x,
                      labels_y=labels_y,
                      title=title,
                      figxsize=figxsize,
                      figysize=figysize,
                      img_filepath=img_filepath,
                      cmap=cmap);
    return MI_XY;

def plot_mutual_info_regression_xy_vector(  X_in,
                                            y_in,
                                            labels_x=None,
                                            title='',
                                            figxsize=15,
                                            figysize=2,
                                            img_filepath=None,
                                            horizontal=False):
    if len(X_in.shape)==1:
        Nx=1;
    else:
        Nx=X_in.shape[1];
    
    if len(y_in.shape)!=1:
        sys.exit('Problem with shape. len(y_in.shape)!=1');
    
    X=X_in.reshape((-1,Nx));
    y=y_in.reshape((-1,1));

    if X.shape[0]!=y.shape[0]:
        sys.exit('Problem with shapes: X_in.shape[0]!=y_in.shape[0]')
    L =X.shape[0];
    
    MAX=mutual_info_regression(y.reshape((-1,1)), y.reshape(-1,));
    
    #print('X.shape:',X.shape)
    #print('y.shape:',y.shape)
    
    mat_xy=mutual_info_regression(X, y.reshape(-1,))/MAX;
    
    if labels_x==None:
        labels_x=['f'+str(n) for n in range(Nx)];
        
    Zdat = sorted(zip(mat_xy.tolist(),labels_x));#, reverse=True
    
    DATA_COLUMNS_SORTED_m = [a for _,a in Zdat];
    Data_SORTED           = [a for a,_ in Zdat];
    
    mat_xy_SORTED_m       = np.array(Data_SORTED).reshape((-1,));
    
    
    plot_bar_vec_xy(  mat_xy_SORTED_m,
                      labels_x=DATA_COLUMNS_SORTED_m,
                      label_y='',
                      title=title,
                      figxsize=figxsize,
                      figysize=figysize,
                      img_filepath=img_filepath,
                      horizontal=horizontal);
    
    return mat_xy_SORTED_m;

################################################################################
################################################################################

import numpy as np
import sys
import matplotlib.pyplot as plt

def plot_features_vs_target(X,y,Ncols=3,fig_height=None,fig_width=15,label_x=None,label_y=None):
    if len(X.shape)!=2:
        sys.exit('Problems, the value len(X.shape) shoul be equald to 2, Current value: '+str(len(X.shape)));
    if len(y.shape)!=1:
        sys.exit('Problems, the value len(y.shape) shoul be equald to 1, Current value: '+str(len(y.shape)));
    if X.shape[0]!=y.shape[0]:
        sys.exit('Problems, the value X.shape[0]!=y.shape[0]');

    L=X.shape[0];
    N=X.shape[1];

    if not isinstance(label_x,list):
        label_x=[];
        for n in range(N):
            label_x.append('x'+str(n));
            
    if not isinstance(label_y,str):
        label_y='y';


    Ncols = int(Ncols)
    Nlins = int(np.ceil(N*1.0/Ncols));

    
    if fig_height is None:
        fig_height=(fig_width*Nlins*1.0)/Ncols;

    fig, axs = plt.subplots(Nlins, Ncols);
    fig.set_figwidth(fig_width);
    fig.set_figheight(fig_height);
    fig.tight_layout(pad=2.0)
    n=0;
    for l in range(Nlins):
        for c in range(Ncols):
            if n<N:
                axs[l, c].scatter(X[:,n], y)
                axs[l, c].set_xlabel(label_x[n])
                if c==0:
                    axs[l, c].set_ylabel(label_y)
                #axs[l, c].set_title(label_x[n]+' vs '+label_y);
                n=n+1;


