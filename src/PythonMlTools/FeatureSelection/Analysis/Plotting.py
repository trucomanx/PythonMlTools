#!/usr/bin/python
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
    
    L =X.shape[0];

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
    print(np.max(MI_X))    
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


