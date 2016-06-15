"""
Driver  for the EI algorithm developed in the pareto class,
to test the various cases for a test case.
This driver has two objectives and six design parameters. 

Authors:  

Date : October 1,  2015
"""
 
import matplotlib
matplotlib.use('PS')
import numpy as np
import design

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fem_wire_problem
from matplot import * # MATPLOT GOES BEFORE ANYTHING ELSE
from pydes import *
import test_functions
from mpl_toolkits.mplot3d import Axes3D
from math import *
import math
import csv
import copy
import shutil



    # These are from Knowles et al. (2005)
np.random.seed(123456)

class ObjFunc(object):



    def f1(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x1 = copy.copy(x)
            xi_1 = self.sigma1*np.random.randn((np.size(x1)/2),)
            xi_2 = self.sigma2*np.random.randn((np.size(x1)/2),)
            xi = np.concatenate((xi_1,xi_2))
            x1 = x1 + xi
            g = 0
            for i in range(1,len(x1),1):
                g = g + (x1[i]-0.5)**2 - np.cos(2*np.pi*x1[i])
            g = 100*(g+5)
            k = 0.5*(x1[0])*(g+1)
            y = y + k
        return float(y)/self.n_samp
        #a = np.array([0,0,1,0,1,0])
        #b = np.array([10,10,5,6,5,10])
        #x_ = (b-a)*x_ + a
        #g = (-25*((x_[0]-1)**2) - (x_[1]-2)**2 - (x_[2]-1)**2 - (x_[3]-4)**2 - (x_[4]-1)**2)
        #y = g
        

    def f2(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x2 = copy.copy(x)
            xi_1 = self.sigma1*np.random.randn((np.size(x2)/2),)
            xi_2 = self.sigma2*np.random.randn((np.size(x2)/2),)
            xi = np.concatenate((xi_1,xi_2))
            x2 = x2 + xi
            #print x2
            #a = np.array([0,0,1,0,1,0])
            #b = np.array([10,10,5,6,5,10])
            #x2 = (b-a)*x2 + a
            #print x2
            g = 0
            for i in range(1,len(x2),1):
                g = g + (x2[i]-0.5)**2 - np.cos(2*np.pi*x2[i])
            g = 100*(g+5)
            k = 0.5*(1-x2[0])*(g+1)
            y = y + k
        return float(y)/self.n_samp

    def __init__(self,sigma1=0.,sigma2=0.,n_samp=1): 
        self.n_samp = n_samp
        self.sigma1 = sigma1
        self.sigma2 = sigma2


    def __call__(self,x):
        return self.f1(x), self.f2(x)
 


if __name__ == '__main__':
    assert len(sys.argv)==3
    noise = float(sys.argv[1])
    n = int(sys.argv[2])

    ObjFunc_noise = ObjFunc(noise,noise,1)
    ObjFunc_true = ObjFunc(noise,noise,100)
    
    # The objective function `dtlz1a`.
    def obj_funcs_noise(x1):
        return ObjFunc_noise.__call__(x1)

    def obj_funcs_true(x1):
        return ObjFunc_true.__call__(x1)
    out_dir = 'ex1_results_n={0:d}_sigma={1:s}'.format(n,sys.argv[1])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    X_init = design.latin_center(n, 6, seed=1234)
    Y_init = np.array([obj_funcs_noise(x) for x in X_init])
    X_d_for_true = design.latin_center(10000, 6, seed=23415)
    X_design = design.latin_center(100, 6, seed=314519)
    Y_true = np.array([obj_funcs_true(x) for x in X_d_for_true])
    #Y_true = np.load('true_data.npy')
    pareto = ParetoFront(X_init, Y_init, obj_funcs_noise, obj_funcs_true,
                         X_design=1000,
                         gp_opt_num_restarts=20,
                         verbose=True,
                         max_it = 100,
                         make_plots=True,
                         add_at_least=30,
                         get_fig=get_full_fig,
                         fig_prefix=os.path.join(out_dir,'ex1'),
                         Y_true_pareto=Y_true,
                         gp_fixed_noise=None,
                         samp=100,
                         denoised=None
                         )     
    pareto.optimize()
