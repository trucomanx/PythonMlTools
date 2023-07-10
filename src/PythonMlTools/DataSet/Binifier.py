#!/usr/bin/python

#import os
#import sys


import numpy as np



class Binifier:

    def __init__(self, X, y,bins=10):
        
        if isinstance(bins, int) and bins>0:
            self.bins=bins;
        else:
            sys.exit('Problems, bins should be a nonzero positive integer. bins='+str(bins));
        
        if len(y.shape)!=1:
            sys.exit('The shape should have len(y.shape)==1, current:',str(len(y.shape)));
        if len(X.shape)!=2:
            sys.exit('The shape should have len(X.shape)==2, current:',str(len(X.shape)));
        
        if y.shape[0] != X.shape[0]:
            sys.exit('Shape of X and y is not compatible.');
        
        L=X.shape[0];
        N=X.shape[1];
        
        self.MIN_X=np.zeros((N,));
        self.MAX_X=np.zeros((N,));
        self.Delta_X=np.zeros((N,));
        for n in range(N):
            self.MIN_X[n]=np.min(X[:,n]);
            self.MAX_X[n]=np.max(X[:,n]);
            self.Delta_X[n]=(self.MAX_X[n]-self.MIN_X[n])/self.bins;
        
        self.MIN_y=np.min(y);
        self.MAX_y=np.max(y);
        self.Delta_y=(self.MAX_y-self.MIN_y)/self.bins;
        
        
    def transform_x(self, X):
        X_new=np.zeros(X.shape);
        L=X.shape[0];
        N=X.shape[1];
        
        for n in range(N):
            for l in range(L):
                if self.Delta_X[n]==0:
                    X_new[l,n]=self.MIN_X[n];
                    
                else:
                    m=np.floor((X[l,n]-self.MIN_X[n])/self.Delta_X[n]);
                    
                    if m<0:
                        m=0;
                    if m>=self.bins:
                        m=self.bins-1;
                    
                    X_new[l,n]= (0.5+m)*self.Delta_X[n] + self.MIN_X[n];
                    
        return X_new;
    
    def transform_y(self, y, index_type=False):
        y_new=np.zeros(y.shape);
        L=y.shape[0];
        
        for l in range(L):
            if self.Delta_y==0:
                y_new[l]=self.MIN_y;
            else:
                m=np.floor((y[l]-self.MIN_y)/self.Delta_y);
                if m<0:
                    m=0;
                if m>=self.bins:
                    m=self.bins-1;
                
                if index_type:
                    y_new[l]= m;
                else:
                    y_new[l]= (0.5+m)*self.Delta_y + self.MIN_y;
        return y_new;
    
    def untransform_y_index_type(self, y_new):
        y=np.zeros(y_new.shape);
        L=y_new.shape[0];
        
        for l in range(L):
            m=y_new[l];
            if m<0:
                m=0;
            if m>=self.bins:
                m=self.bins-1;
            
            y[l]= (0.5+m)*self.Delta_y + self.MIN_y;
        return y;
        
