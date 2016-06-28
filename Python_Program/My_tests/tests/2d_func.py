"""
Objective function script.
"""

import matplotlib
matplotlib.use('PS')
import numpy as np
import design
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from matplot import * # MATPLOT GOES BEFORE ANYTHING ELSE
from pydes import *

from mpl_toolkits.mplot3d import Axes3D
from math import *
import math
import csv
import copy
from ast import literal_eval

np.random.seed(123458)

class ObjFunc(object):

    def f1(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x1 = copy.copy(x)
            xi = self.sigma*np.random.randn(np.size(x1),)
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
            xi = self.sigma*np.random.randn(np.size(x2),)
            x2 = x2 + xi
            b1 = 15. * x2[0] - 5.
            b2 = 15. * x2[1]
            k = - np.sqrt(np.abs((10.5 - b1) * ((b1 + 5.5)) * (b2 + 0.5))) \
            - 1. / 30. * (b2 - 5.1 / 4. / math.pi ** 2 * b1 ** 2 - 6.) ** 2 \
            - 1. / 3. * ((1. - 1. / 8. / math.pi) * math.cos(b1) + 1.)
            y = y + k
        return float(y)/self.n_samp

    def __init__(self,sigma=0.,n_samp=1.):
        self.n_samp = n_samp
        self.sigma = sigma

    def __call__(self,x):
        return self.f1(x), self.f2(x)

if __name__ == '__main__':
    noise = float(sys.argv[1])
    ObjFunc_noise = ObjFunc(noise,1)

    # The objective function from `Binois et al`.
    def obj_funcs_noise(x):
        return ObjFunc_noise.__call__(x)

    if sys.argv[2].isdigit():
        x = design.latin_center(sys.argv[2],2)
        x_string_array = str(x).split('\n')
        inp = open('x.csv','wb')
        for i, string in enumerate(x_string_array):
            n_string = string.replace('[','')
            n_string = n_string.replace(']','')
            n_string = n_string.replace(' ','')
            if i < len(x_string_array)-1:
                inp.write(n_string[:4] + ',' + n_string[4:len(n_string)] + '\n')
            else:
                inp.write(n_string[:4] + ',' + n_string[4:len(n_string)])
        inp.close()

        print 'the inputs are'+ str(x)
        y = np.array([obj_funcs_noise(x) for x in x])

        out = open('y.csv','wb')
        for i, data in enumerate(y):
            if i < len(y) - 1:
                out.write(str(data[0]) + ',' + str(data[1]) + '\n')
            else:
                out.write(str(data[0]) + ',' + str(data[1]))
        out.close()

        #Open everything
        inp = open('x.csv','rb')
        line = inp.read()
        x_data = list(line.split('\n'))
        data_list = []
        for i in range(len(x_data)):
            print x_data[i].split(',')
            data_list.append(tuple(map(float,x_data[i].split(','))))
        t = tuple(data_list)
        print t

        out = open('y.csv','rb')
        line = out.read()
        y_data = list(line.split('\n'))
        data_list = []
        for i in range(len(y_data)):
            print y_data[i].split(',')
            data_list.append(tuple(map(float,y_data[i].split(','))))
        t_y = tuple(data_list)
        print t_y



    else:
        for i in sys.argv:
            print i
        x_d = []
        #x_d = sys.argv[2:]
        for i in sys.argv[2:]:
            x_d.append(float(i))

        y = obj_funcs_noise(x_d)

    print 'the corresponding outputs are'+ str(y)
