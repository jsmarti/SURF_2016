"""
Test the heteroscedastic GP of GPy.

Author:
    Ilias Bilioinis

Date:
    5/22/2015

"""


import GPy
import os
import numpy as np
import scipy
from pydes import HeteroscedasticGPRegression, GPRegression
import pymc as pm
import matplotlib.pyplot as plt 
import seaborn as sns
import math


if __name__ == '__main__':
    #motor_data_file = os.path.join(os.path.dirname(__file__), 'motor.dat')
    #motor_data_mfile = os.path.join(os.path.dirname(__file__), 'motorcycle.mat')
    #data = scipy.io.loadmat(motor_data_mfile)
    #X = data['X'][::5, :]
    #Y = data['y'][::5, :]
    #Xd = np.linspace(0, 70, 100)[:, None]
    #X = (X - np.mean(X)) / np.std(X)
    #Y = (Y - np.mean(Y)) / np.std(Y)
    X = np.random.rand(100, 1)
    Y = 2. * np.sin(2. * math.pi * X) + \
        (.5 + np.abs(np.cos(2 * math.pi * X))) * np.random.randn(X.shape[0], 1)
    plt.plot(X, Y, '.')
    plt.plot(X, (.5 + np.exp(X) * np.abs(np.cos(2 * math.pi * X))), '.')
    plt.show()
    Xd = np.linspace(0, 1, 100)[:, None]
    model = HeteroscedasticGPRegression(X, Y)
    mcmc = model.sample(100000, burn=30000, thin=1000,
                        tune_interval=100)
    pm.Matplot.plot(mcmc)
    plt.show()
    g = mcmc.trace('g')[-1, :]
    plt.plot(X, g, '.')
    plt.plot(X, (.5 + np.abs(np.cos(2 * math.pi * X))), '.')
    #plt.fill_between(Xd.flatten(), m_m - 2. * m_s, m_m + 2. * m_s, color='grey', alpha=.25)
    #plt.plot(Xd, m_s)
    plt.show()
    quit()
    graph = pm.graph.graph(model)
    graph.write_png('test.png')
    quit()
    #print str(model)
    model.optimize(method='l-bfgs-b')
    print str(model)
    #print str(model)
    x = np.linspace(X.min(), X.max(), 100)[:, None]
    out1, out2, mutst,diagSigmatst,atst,diagCtst = model.predict(x)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(X, Y, '.')
    ax1.plot(x, atst)
    l = atst.flatten() - 2. * np.sqrt(diagCtst).flatten()
    u = atst.flatten() + 2. * np.sqrt(diagCtst).flatten()
    ax1.fill_between(x.flatten(), l, u, color='green', alpha=0.25)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, mutst)
    lg = mutst.flatten() - 2. * np.sqrt(diagSigmatst).flatten()
    ug = mutst.flatten() + 2. * np.sqrt(diagSigmatst).flatten()
    ax2.fill_between(x.flatten(), lg, ug, color='blue', alpha=.25)
    plt.show()
    a = raw_input('pe')
    #J, dJ = model.objective_function()
    #x0 = model.optimizer_array
    #print J
    #func = lambda(x): model.objective_function(x)[0]
    #grad = scipy.optimize.approx_fprime(x0, func, 1e-6)
    #print dJ - grad