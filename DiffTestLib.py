# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:14:00 2015
various support function for evaluation
@author: eugene
"""
import scipy.stats
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model.base import center_data
from sklearn.utils import check_random_state
import numpy as np
import scipy as sp

def generate_sparse_gaussian_graph_with_binary_confound3(n_features=30,edge_changes=3,n_samples=30, min_n_samples=30,max_n_samples=50,
        density=0.05,random_state=0, verbose=0,pin=False):
  

    random_state = check_random_state(random_state)
    density*=0.5
    # Generate topology (upper triangular binary matrix, with zeros on the
    # diagonal)
    topology = np.empty((n_features, n_features))
    topology[:, :] = np.triu((
        random_state.randint(0, high=int(1. / density),
                             size=n_features * n_features)
    ).reshape(n_features, n_features) == 0, k=1)

    edges=np.where(topology>0)
    rem=np.random.choice(edges[0].shape[0],edge_changes,replace=False)
    topology2=topology.copy()
    topology2[edges[0][rem],edges[1][rem]]=0
    # Generate edges weights on topology
    precisions = []
    mask = topology > 0
    # See also sklearn.datasets.samples_generator.make_sparse_spd_matrix
    prec = topology.copy()
    prec[mask] = random_state.uniform(low=.5, high=.5, size=(mask.sum()))
    #prec += np.eye(prec.shape[0])
    prec2=prec.copy()
    prec2[edges[0][rem[0:len(rem)/2]],edges[1][rem[0:len(rem)/2]]]=0
    prec2[edges[1][rem[0:len(rem)/2]],edges[0][rem[0:len(rem)/2]]]=0
    prec[edges[0][rem[len(rem)/2:]],edges[1][rem[len(rem)/2:]]]=0
    prec[edges[1][rem[len(rem)/2:]],edges[0][rem[len(rem)/2:]]]=0


    prec=prec.T+prec
    prec2=prec2.T+prec2
    c1=prec.sum(axis=1)
    c1[c1==0]=1.
    c2=prec2.sum(axis=1)
    c2[c2==0]=1.
    
    prec=prec/(c1*c2)
    prec=prec.T+prec    
        
    prec2=prec2/(c1*c2)
    prec2=prec2.T+prec2
    
    
    prec2+=np.eye(prec2.shape[0])
    prec+=np.eye(prec.shape[0])
    


   # prec2[np.diag_indices_from(prec2)]=prec.diagonal() #make sure the diagonal doesnt have zeros
    # Assert precision matrix is spd
    np.testing.assert_almost_equal(prec, prec.T)
    np.testing.assert_almost_equal(prec2, prec2.T)
    eigenvalues = np.linalg.eigvalsh(prec)
    if eigenvalues.min() < 0:
        raise ValueError("Failed generating a positive definite precision "
                         "matrix. Decreasing n_features can help solving "
                         "this problem.")
    eigenvalues = np.linalg.eigvalsh(prec2)
    if eigenvalues.min() < 0:
        raise ValueError("Failed generating a positive definite precision "
                         "matrix. Decreasing n_features can help solving "
                         "this problem.")

#    # Returns the topology matrix of precision matrices.
    topology += np.eye(*topology.shape)
    topology = np.dot(topology.T, topology)
    topology = topology > 0

    topology2 += np.eye(*topology2.shape)
    topology2 = np.dot(topology2.T, topology2)
    topology2 = topology2 > 0
    precisions.append(prec)
    precisions.append(prec2)
    # Generate temporal signals
    signals = generate_signals_from_precisions(precisions,min_n_samples=min_n_samples, max_n_samples=max_n_samples,
                                               random_state=random_state,pin=pin)
                                        
    return signals, precisions, topology,topology2

def generate_signals_from_precisions(precisions,
                                     min_n_samples=50, max_n_samples=100,
                                     random_state=0,pin=False):
    """Generate timeseries according to some given precision matrices.

    Signals all have zero mean.

    Parameters
    ----------
    precisions: list of numpy.ndarray
        list of precision matrices. Every matrix must be square (with the same
        size) and positive definite. The output of
        generate_group_sparse_gaussian_graphs() can be used here.

    min_samples, max_samples: int
        the number of samples drawn for each timeseries is taken at random
        between these two numbers.

    Returns
    -------
    signals: list of numpy.ndarray
        output signals. signals[n] corresponds to precisions[n], and has shape
        (sample number, precisions[n].shape[0]).
    """
    random_state = check_random_state(random_state)

    signals = []
    n_samples = random_state.randint(min_n_samples, high=max_n_samples,
                                     size=len(precisions))

    mean = np.zeros(precisions[0].shape[0])
    for n, prec in zip(n_samples, precisions): 
        if(pin):
            signals.append(random_state.multivariate_normal(mean,
                                                    np.linalg.pinv(prec),
                                                    (n,)))
        else:
            signals.append(random_state.multivariate_normal(mean,
                                                    np.linalg.inv(prec),
                                                    (n,)))
    return signals

def plot_matrix(m, ylabel=""):
    abs_max = abs(m).max()
    plt.imshow(m, cmap=plt.cm.RdBu_r, interpolation="nearest",
              vmin=-abs_max, vmax=abs_max)


def Type_I_II(Estimated_Topology,True_Topology):
    E_est=np.abs(Estimated_Topology)>0 #Reject H0 
    E_t=np.abs(True_Topology)>0 #H1 True
    FP= float((np.logical_and(E_est,~E_t)).sum())  
        
    FN=float((np.logical_and(~E_est,E_t)).sum())
    return FP,FN

def Power(Pvals,alpha,H1_True):  
    Rejected=Pvals<alpha
    Power=float(Rejected[H1_True].sum())/float(H1_True.sum())
    return Power

def Coverage_Length_CI(Upper,Lower,H1_True,True_Prec):
    E_n=float(H1_True.sum())
    E_c_n=float((~H1_True).sum())
    
        
    if(E_c_n==0):
        coverage_s0=0
        length_s0=0
    else:
        coverage_s0=float(np.logical_and(True_Prec[~H1_True]<Upper[~H1_True],True_Prec[~H1_True]>Lower[~H1_True]).sum())/E_c_n
        length_s0=(Upper[~H1_True]-Lower[~H1_True]).mean()
        
    if(E_n==0):
        coverage_s0_c=0
        length_s0_c=0
    else:
        coverage_s0_c=float(np.logical_and(True_Prec[H1_True]<Upper[H1_True],True_Prec[H1_True]>Lower[H1_True]).sum())/E_n
        length_s0_c=(Upper[H1_True]-Lower[H1_True]).mean()
    return coverage_s0,coverage_s0_c,length_s0,length_s0_c