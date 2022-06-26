#!/usr/bin/python
import numpy as np

def kde_get2d_joint_prob(kde, min_x,max_x,bins_x,min_y,max_y,bins_y):
    spacex=np.linspace(min_x,max_x,bins_x);
    spacey=np.linspace(min_y,max_y,bins_y);
    
    xv, yv = np.meshgrid(spacex, spacey, indexing='ij');
    
    xv=xv.reshape((xv.shape[0]*xv.shape[1],1));#column vector
    yv=yv.reshape((yv.shape[0]*yv.shape[1],1));#column vector
    
    XY=np.concatenate((xv, yv), axis=1);
    
    Pab=np.exp(kde.score_samples(XY));
    
    Pab=np.reshape(Pab,(bins_x,bins_y));
    Pab=Pab/np.sum(Pab);
    return Pab;
