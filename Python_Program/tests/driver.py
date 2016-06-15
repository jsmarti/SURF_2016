"""
Driver  for the EI algorithm developed in the pareto class,
to test the various cases for the wire drawing problem.
This driver has two objectives and two design parameters. 

Authors: Ilias Bilionis, Piyush Pandita 

Date : 23 June 2015
"""
 
import matplotlib
matplotlib.use('PS')
import numpy as np
import design
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from matplot import * # MATPLOT GOES BEFORE ANYTHING ELSE
import fem_wire_problem
from pydes import *
import test_functions
from mpl_toolkits.mplot3d import Axes3D
from math import *
import math
import csv

if __name__ == '__main__':

    # These are from Binois et al. (2015)
    np.random.seed(123456)
   
   
    def f1(x):
	g = 0
    	for i in range(1,len(x),1):
            g = g + (x[i]-0.5)**2 - np.cos(2*np.pi*x[i])
    	g = 100*(g+5)
    	y = 0.5*(x[0])*(g+1)
    	return y

    def f2(x):
	g = 0
    	for i in range(1,len(x),1):
            g = g + (x[i]-0.5)**2 - np.cos(2*np.pi*x[i])
    	g = 100*(g+5)
    	y = 0.5*(1-x[0])*(g+1)
    	return y
    
    # The instance of the function evaluation wrapper class. enables creation 
    # of separate input files to be used later on 
    wire_pass1 = fem_wire_problem.FunctionEvaluationWrapper(0.0,0.1,1)
    wire_pass1_true = fem_wire_problem.FunctionEvaluationWrapper(0.0,0.1,10)
    # I changed the objective functions to function like this:
    def obj_funcs(x):
	#return f1(x),f2(x)
        return wire_pass1.__call__(x)

    def obj_funcs_true(x):
	   return wire_pass1_true.__call__(x)

    Y_true = np.load('fewire_15d_1000_output.npy') 
    #y1 = data[:,0]
    #y2 = data[:,1]
    #y1.reshape((1000,1))
    #y2.reshape((1000,1))
    #k1 = (48)*np.ones((1000,1))
    #k2 = (6.4)*np.ones((1000,1))
    #y1 = (y1-k1)/2
    #y2 = (y2-k2)
    #Y_true = np.hstack([y1,y2]) # The array with the observed values of the objectives
    X_init = design.latin_center(15, 15, seed=1234)
    Y_init = np.array([obj_funcs(x) for x in X_init])
    X_d_for_true = design.latin_center(10000, 15, seed=23415)
    X_design = design.latin_center(1000, 15, seed=314519)
    #Y_true = np.array([true_obj_funcs(x) for x in X_d_for_true])
    #fig, ax = plot_pareto(Y_true)
    #fig.savefig('sample_pareto.pdf')
    pareto = ParetoFront(X_init, Y_init, obj_funcs,
			 obj_funcs_true,
                         X_design=1000,
                         gp_opt_num_restarts=20,
                         verbose=True,
                         make_plots=True, 
                         get_fig=get_full_fig,
                         max_it = 50,
                         fig_prefix='15var_8pass',
                         Y_true_pareto=Y_true,
                         denoised=True,
                         gp_fixed_noise=None,
                         samp=20) 
    pareto.optimize()
    np.save('output_'+str(pareto.fig_prefix)+'_.npy',pareto.Y_pareto)
    np.save('input_'+str(pareto.fig_prefix)+'_.npy',pareto.X_pareto)
