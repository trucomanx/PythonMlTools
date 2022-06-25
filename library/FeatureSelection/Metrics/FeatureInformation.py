#!/usr/bin/python

import numpy as np


'''
Input a: list or numpy array. Need the same size of b.
Input b: list or numpy array. Need the same size of a.
Input bins_a: Integer number. It doesn't need to be the same size as bins_b.
Input bins_b: Integer number. It doesn't need to be the same size as bins_a.

Return mutual: Real value. Mutual information between a discretized version of a (with bins_a) and b (with bins_b).
Return joint: Real value. Joint information of a discretized version of a (with bins_a) and b (with bins_b).
Return infa: Real value. Information of a discretized version of a (with bins_a).
Return infb: Real value. Information of a discretized version of a (with bins_a).

'''
def AllInformation(a,b,bins_a=2,bins_b=2):
    a=np.asarray(a);
    b=np.asarray(b);
    
    a = np.reshape(a, -1);
    b = np.reshape(b, -1);
    
    assert bins_a>=2, "The bins number of array \"a\" should be greather than 1.";
    assert bins_b>=2, "The bins number of array \"b\" should be greather than 1.";
    assert len(a) == len(b), "The length of both arrays is not the same.";
    
    hab=np.zeros((bins_a, bins_b));
    
    mina=np.min(a);    maxa=np.max(a);  factora=(bins_a-1)/(maxa-mina);
    minb=np.min(b);    maxb=np.max(b);  factorb=(bins_b-1)/(maxb-minb);
    
    a=factora*(a-mina);
    b=factorb*(b-minb);
    
    for n in range(len(a)):
        ida=int(np.round(a[n]));
        idb=int(np.round(b[n]));
        hab[ida][idb]=hab[ida][idb]+1;
    
    ha=np.zeros((bins_a, 1));
    for n in range(len(a)):
        ida=int(np.round(a[n]));
        ha[ida][0]=ha[ida][0]+1;
    
    hb=np.zeros((bins_b, 1));
    for n in range(len(b)):
        idb=int(np.round(b[n]));
        hb[idb][0]=hb[idb][0]+1;
    
    ha=ha/np.sum(ha);
    hb=hb/np.sum(hb);
    hab=hab/np.sum(hab);
    
    #print(ha)
    #print(hb)
    #print(hab)
    
    joint=0.0;
    mutual=0.0;
    for na in range(bins_a):
        for nb in range(bins_b):
            if hab[na][nb]!=0 :
                joint=joint-hab[na][nb]*np.log2(hab[na][nb]);
                mutual=mutual+hab[na][nb]*np.log2(hab[na][nb]/(ha[na][0]*hb[nb][0]));
    
    infa=0.0;
    for na in range(bins_a):
        if ha[na][0]!=0 :
            infa=infa-ha[na][0]*np.log2(ha[na][0]);
    
    infb=0.0;
    for nb in range(bins_b):
        if hb[nb][0]!=0 :
            infb=infb-hb[nb][0]*np.log2(hb[nb][0]);
    
    return mutual,joint,infa,infb;
    
    
