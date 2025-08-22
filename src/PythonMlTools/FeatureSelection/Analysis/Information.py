#!/usr/bin/python

import numpy as np
from sklearn.neighbors import KernelDensity
import PythonMlTools.DensityEstimation.KdeExtras as kext
import PythonMlTools.FeatureSelection.Metrics.FeatureInformation as FI
import sys

def all_against_all_mutual_inf( X,
                                bins=10,
                                bandwidth=0.08,
                                kernel_type='gaussian',
                                IoU=True):
    '''
    Lee un archivo de imagen `filepath` y retorna las anotaciones y la imagen leida.

    :param X: Matriz de datos con muestras en las lineas.
    :type X: numpy array
    :param bins: Número de partes para discretizar la función de densidad de probabilidad. Se recomienda bins <sqrt(X.Nlins).
    :type bins: int
    :param kernel_type: Tipo de kernel density estimation (sklearn.neighbors.KernelDensity).
    :type kernel_type: str
    :param bandwidth: Parámetro bandwidth del generador de kde (sklearn.neighbors.KernelDensity). Se recomienda menor a 1/bins (bandwidth < 1.0/bins).
    :type bandwidth: float
    :return: Retorna una matriz cuadrada (Ncol x Ncol) con la información mutua de todos contra todos.
    :rtype: numpy array
    '''
    
    if(len(X.shape)<2):
        sys.exit('The length shape of X should be 2. Current len(X.shape)='+str(len(X.shape)));
    
    L=X.shape[0];
    N=X.shape[1];
    mat=np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            # Primer vector normalizado
            Xn=X[:,n];
            Xn=Xn-np.sum(Xn);
            STD=np.std(Xn);
            if(STD!=0):
                Xn=Xn/STD;
            
            # Segundo vector normalizado
            Xm=X[:,m];
            Xm=Xm-np.sum(Xm);
            STD=np.std(Xm);
            if(STD!=0):
                Xm=Xm/STD;
            
            xx=np.concatenate((np.reshape(Xn,(L,1)),np.reshape(Xm,(L,1))), axis=1)
            kde = KernelDensity(kernel=kernel_type, bandwidth=bandwidth).fit(xx);

            Pab=kext.kde_get2d_joint_prob(  kde,
                                            np.min(Xn),np.max(Xn),bins,
                                            np.min(Xm),np.max(Xm),bins);

            InfA=FI.InformationAnalysis(Pab);
            mutual=InfA.MutualInformation();
            
            if IoU:
                joint=InfA.JointEntropy();
                mat[n][m]=mutual/joint;
            else:
                mat[n][m]=mutual;
    
    if IoU:
        return mat
    else:
        MAX=np.max(mat);
        if MAX==0:
            MAX=1;
        
        return mat/MAX;


def x_against_y_mutual_inf( X,
                            Y,
                            bins=10,
                            bandwidth=0.08,
                            kernel_type='gaussian',
                            IoU=True):
    '''
    Lee un archivo de imagen `filepath` y retorna las anotaciones y la imagen leida.

    :param X: Matriz de datos con muestras en las lineas.
    :type X: numpy array
    :param Y: Matriz de datos con muestras en las lineas.
    :type Y: numpy array
    :param bins: Número de partes para discretizar la función de densidad de probabilidad. Se recomienda bins <sqrt(X.Nlins).
    :type bins: int
    :param kernel_type: Tipo de kernel density estimation (sklearn.neighbors.KernelDensity).
    :type kernel_type: str
    :param bandwidth: Parámetro bandwidth del generador de kde (sklearn.neighbors.KernelDensity). Se recomienda menor a 1/bins (bandwidth < 1.0/bins).
    :type bandwidth: float
    :return: Retorna una matriz cuadrada (Ncol x Ncol) con la información mutua de todos contra todos.
    :rtype: numpy array
    '''
    if(X.shape[0]!=Y.shape[0]):
        sys.exit('Number of elements of inputs is not the same.');
    
    if(len(X.shape)<2):
        sys.exit('The length shape of X should be 2. Current len(X.shape)='+str(len(X.shape)));
    if(len(Y.shape)<2):
        sys.exit('The length shape of Y should be 2. Current len(Y.shape)='+str(len(Y.shape)));
    
    L=X.shape[0];
    N=X.shape[1];
    M=Y.shape[1];
    mat=np.zeros((N, M))
    for m in range(M):
        
        # Segundo vector normalizado
        Ym=Y[:,m];
        Ym=Ym-np.sum(Ym);
        STD=np.std(Ym);
        if(STD!=0):
            Ym=Ym/STD;
        
        infY = None
        if not IoU:
            yy=np.concatenate((np.reshape(Ym,(L,1)),np.reshape(Ym,(L,1))), axis=1);
            kde = KernelDensity(kernel=kernel_type, bandwidth=bandwidth).fit(yy);

            Pyy=kext.kde_get2d_joint_prob(  kde,
                                            np.min(Ym),np.max(Ym),bins,
                                            np.min(Ym),np.max(Ym),bins);
            Pyy=(Pyy+Pyy.T)/2.0;
            infY=FI.InformationAnalysis(Pyy).MutualInformation();
        
        for n in range(N):
            
            # Primer vector normalizado
            Xn=X[:,n];
            Xn=Xn-np.sum(Xn);
            STD=np.std(Xn);
            if(STD!=0):
                Xn=Xn/STD;
            
            
            xy=np.concatenate((np.reshape(Xn,(L,1)),np.reshape(Ym,(L,1))), axis=1)
            kde = KernelDensity(kernel=kernel_type, bandwidth=bandwidth).fit(xy);

            Pab=kext.kde_get2d_joint_prob(  kde,
                                            np.min(Xn),np.max(Xn),bins,
                                            np.min(Ym),np.max(Ym),bins);

            InfA=FI.InformationAnalysis(Pab);
            mutual=InfA.MutualInformation();
            
            if IoU:
                joint=InfA.JointEntropy();
                mat[n][m]=mutual/joint;
            else:
                mat[n][m]=mutual/infY;
    return mat;



