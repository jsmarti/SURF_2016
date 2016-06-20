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



    # These are from Knowles et al. (2005)
np.random.seed(123458)

class ObjFunc(object):

    def f1(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x1 = copy.copy(x)
            xi_1 = self.sigma1*np.random.randn((np.size(x1)/2),)
            xi_2 = self.sigma2*np.random.randn((np.size(x1)/2),)
            xi = np.concatenate((xi_1,xi_2))
            x1 = x1 + xi
            b1 = 15. * x1[0] - 5.
            b2 = 15. * x1[1]
            k = (b2 - 5.1 / 4. / math.pi ** 2 * b1 ** 2 + 5. / math.pi * b1 - 6.) ** 2. \
            + 10. * ((1. - 1. / 8. / math.pi) * math.cos(b1) + 1.)
            y = y + k    
        return float(y)/self.n_samp

    def f2(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x2 = copy.copy(x)
            xi_1 = self.sigma1*np.random.randn((np.size(x2)/2),)
            xi_2 = self.sigma2*np.random.randn((np.size(x2)/2),)
            xi = np.concatenate((xi_1,xi_2))
            x2 = x2 + xi
            b1 = 15. * x2[0] - 5.
            b2 = 15. * x2[1]
            k = - np.sqrt(np.abs((10.5 - b1) * ((b1 + 5.5)) * (b2 + 0.5))) \
            - 1. / 30. * (b2 - 5.1 / 4. / math.pi ** 2 * b1 ** 2 - 6.) ** 2 \
            - 1. / 3. * ((1. - 1. / 8. / math.pi) * math.cos(b1) + 1.)
            y = y + k
        return float(y)/self.n_samp

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
    
    out_dir = 'ex1_results_n={0:d}_sigma={1:s}'.format(n,sys.argv[1])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    # The objective function from `Binois et al`.
    def obj_funcs_noise(x1):
        return ObjFunc_noise.__call__(x1)

    def obj_funcs_true(x1):
        return ObjFunc_true.__call__(x1)
    
    X_init = design.latin_center(n, 2, seed=1234)
    Y_init = np.array([obj_funcs_noise(x) for x in X_init])
    X_d_for_true = design.latin_center(1000, 2, seed=23415)
    X_design = design.latin_center(100, 2, seed=314519)
    Y_true = np.array([obj_funcs_true(x) for x in X_d_for_true])
    #np.save('ex2_data1',Y_true)
    #quit()
    #Y_true = np.load('ex2_data1.npy')
    pareto = ParetoFront(X_init, Y_init, obj_funcs_noise, obj_funcs_true,
                         X_design=100,
                         gp_opt_num_restarts=20,
                         verbose=True,
                         max_it=100,
                         make_plots=True,
                         add_at_least=30,
                         get_fig=get_full_fig,
                         fig_prefix=os.path.join(out_dir,'ex1'),
                         Y_true_pareto=Y_true,
			             gp_fixed_noise=None,
                         denoised=True,
                         samp=100)     
    pareto.optimize()
