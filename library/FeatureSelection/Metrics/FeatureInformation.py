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
class InformationAnalysis:

        
        
    def __init__(self, *args):
        self.bins_a=0;
        self.bins_b=0;
        self.Pa=0;
        self.Pb=0;
        self.Pab=0;
        if len(args) == 4:
            self.SetProbabilitiesFromHistogram(args[0], args[1], args[2], args[3]);
        elif len(args) == 1:
            self.SetProbabilities(args[0]);
        else:
            self.bins_a=0;
            self.bins_b=0;
            
        
    def SetProbabilitiesFromHistogram(self, a, b, bins_a, bins_b):
        a=np.asarray(a);
        b=np.asarray(b);
        
        a = np.reshape(a, -1);
        b = np.reshape(b, -1);
        
        assert bins_a>=2, "The bins number of array \"a\" should be greather than 1.";
        assert bins_b>=2, "The bins number of array \"b\" should be greather than 1.";
        assert len(a) == len(b), "The length of both arrays is not the same.";
        
        mina=np.min(a);    maxa=np.max(a);
        minb=np.min(b);    maxb=np.max(b);  
        
        if maxa!=mina:
            a=((bins_a-1)/(maxa-mina))*(a-mina);
        else:
            a=a*0.0;
        
        if maxb!=minb:
            b=((bins_b-1)/(maxb-minb))*(b-minb);
        else:
            b=b*0.0;
        
        self.bins_a=bins_a;
        self.bins_b=bins_b;
        
        # Generating the joint probability self.Pab
        self.Pab=np.zeros((self.bins_a, self.bins_b));
        for n in range(len(a)):
            ida=int(np.round(a[n]));
            idb=int(np.round(b[n]));
            self.Pab[ida][idb]=self.Pab[ida][idb]+1;
        self.Pab=self.Pab/np.sum(self.Pab);

        # Generating the probability self.Pa
        self.Pa=np.reshape(np.zeros((self.bins_a, 1)),-1);
        for n in range(len(a)):
            ida=int(np.round(a[n]));
            self.Pa[ida]=self.Pa[ida]+1;
        self.Pa=self.Pa/np.sum(self.Pa);

        # Generating the probability self.Pb
        self.Pb=np.reshape(np.zeros((bins_b, 1)),-1);
        for n in range(len(b)):
            idb=int(np.round(b[n]));
            self.Pb[idb]=self.Pb[idb]+1;
        self.Pb=self.Pb/np.sum(self.Pb);
        
    def SetProbabilities(self, Pab):
        self.Pab=np.asarray(Pab);
        self.Pab=np.abs(self.Pab);
        self.Pab=self.Pab/np.sum(self.Pab);
        
        self.bins_a, self.bins_b = self.Pab.shape;
        
        self.Pa=np.reshape(np.zeros((self.bins_a, 1)),-1);
        self.Pb=np.reshape(np.zeros((self.bins_b, 1)),-1);
        
        for na in range(self.bins_a):
            for nb in range(self.bins_b):
                self.Pa[na]=self.Pa[na]+self.Pab[na][nb];
                self.Pb[nb]=self.Pb[nb]+self.Pab[na][nb];

    def MutualInformation(self):
        # Mutual information
        mutual=0.0;
        for na in range(self.bins_a):
            for nb in range(self.bins_b):
                if self.Pab[na][nb]!=0 :
                    mutual=mutual+self.Pab[na][nb]*np.log2(self.Pab[na][nb]/(self.Pa[na]*self.Pb[nb]));
        
        return mutual;
        
    def JointEntropy(self):
        # Joint information
        joint=0.0;
        for na in range(self.bins_a):
            for nb in range(self.bins_b):
                if self.Pab[na][nb]!=0 :
                    joint=joint-self.Pab[na][nb]*np.log2(self.Pab[na][nb]);
        
        return joint;
        
    def EntropyA(self):
        # Information of a
        infa=0.0;
        for na in range(self.bins_a):
            if self.Pa[na]!=0 :
                infa=infa-self.Pa[na]*np.log2(self.Pa[na]);
        
        return infa;
        
    def EntropyB(self):
        # Information of b
        infb=0.0;
        for nb in range(self.bins_b):
            if self.Pb[nb]!=0 :
                infb=infb-self.Pb[nb]*np.log2(self.Pb[nb]);
        
        return infb;
    
    
