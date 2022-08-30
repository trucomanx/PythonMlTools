#!/usr/bin/python

import numpy as np
from sklearn.neighbors import KernelDensity
import DensityEstimation.KdeExtras as kext
import FeatureSelection.Metrics.FeatureInformation as FI

def all_against_all_mutual_inf(X,bins=10,bandwidth=0.08,kernel_type='gaussian'):
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
            joint=InfA.JointEntropy();

            mat[n][m]=mutual/joint;
    return mat;
