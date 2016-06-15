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



# The KURSAWE FUNCTION
np.random.seed(123458)

class ObjFunc(object):

    def f1(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x1 = copy.copy(x)
            xi = self.sigma1*np.random.randn((np.size(x1)),)
            x1 = x1 + xi
            a = np.array([-5,-5,-5])
            b = np.array([5,5,5])
            x1 = (b-a)*x1 + a
            y = y + (-10)*(np.exp(-0.2*(np.sqrt(((x1[0])**2) + ((x1[1]**2)))))) + (-10)*(np.exp(-0.2*(np.sqrt(((x1[1])**2) + ((x1[2]**2))))))
        return (float(y)/self.n_samp + 14)/2.

    def f2(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x2 = copy.copy(x)
            xi = self.sigma1*np.random.randn((np.size(x2)),)
            x2 = x2 + xi
            a = np.array([-5,-5,-5])
            b = np.array([5,5,5])
            x2 = (b-a)*x2 + a
            y = y + ((abs(x2[0]))**0.8 + (abs(x2[1]))**0.8 + (abs(x2[2]))**0.8) + (5. *((math.sin((x2[0])**3)) + (math.sin((x2[1])**3)) + (math.sin((x2[2])**3))))
        return (float(y)/self.n_samp + 6)/2.

    def __init__(self,sigma1=0.,sigma2=0.,n_samp=1.): 
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
    
    # The objective function from `Binois et al`.
    def obj_funcs_noise(x1):
        return ObjFunc_noise.__call__(x1)

    def obj_funcs_true(x1):
        return ObjFunc_true.__call__(x1)
    
    X_init = design.latin_center(n, 3, seed=1234)
    Y_init = np.array([obj_funcs_noise(x) for x in X_init])
    X_d_for_true = design.latin_center(10000, 3, seed=23415)
    X_design = design.latin_center(1000, 3, seed=314519)
    Y_true = np.array([obj_funcs_true(x) for x in X_d_for_true])
    out_dir = 'ex3_results_n={0:d}_sigma={1:s}'.format(n,sys.argv[1])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    pareto = ParetoFront(X_init, Y_init, obj_funcs_noise, obj_funcs_true,
                         X_design=1000,
                         gp_opt_num_restarts=40,
                         verbose=True,
                         max_it=50,
                         make_plots=True,
                         add_at_least=30,
                         gp_fixed_noise=None,
                         get_fig=get_full_fig,
                         fig_prefix=os.path.join(out_dir,'ex3'),
                         Y_true_pareto=Y_true,
                         denoised=True,
                         samp=100)     
    pareto.optimize()
