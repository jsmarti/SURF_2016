"""
The plots related to the data set accumulated for pass1 of the wire problem.

"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = np.loadtxt('main_file.csv',delimiter=",",usecols=(0,1,2,3),unpack=True)
x1 = a[0][:,None]  # INPUT 1 Incoming wire diameter
x2 = a[1][:,None]  # INPUT 2 Die Angle
y1 = a[2][:,None]
y2 = (-1)*a[3][:,None] # The data file has the UTS positive so it is being converted here 


k1 = 10*np.ones((1148,1))
k2 = (-1325)*np.ones((1148,1))

y1 = (y1-k1)/2     # Scaling the objective 1
y2 = (y2-k2)/10    # Scaling the objective 2

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1,x2,y1,s=30)
ax.set_xlabel('the incoming wire diameter')
ax.set_ylabel('the die angle')
ax.set_zlabel('the ideal work')

plt.savefig('obj1_3d_plot.pdf')
plt.show(block=True)
plt.clf()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1,x2,y2,s=30)
ax.set_xlabel('the incoming wire diameter')
ax.set_ylabel('the die angle')
ax.set_zlabel('the UTS')

plt.savefig('obj2_3d_plot.pdf')
plt.show(block=True)
plt.clf()




index = np.linspace(0,1148,1148)

"""
y1 = np.array(a[2])[:,None]
y2 = np.array(a[3])[:,None]
y2 = y2*(-1)
k1 = (10)*np.ones((1148,1))
k2 = (-1325)*np.ones((1148,1))
y1 = (y1-k1)/2
y2 = (y2-k2)/10

plt.scatter(index,y1,color='blue',s=30)
plt.title('The Ideal_Work')
plt.savefig('std_norm_obj1.pdf')
plt.show(block=True)
plt.scatter(index,y2,color='blue',s=30)
plt.title('The UTS')
plt.savefig('std_norm_obj2.pdf')
plt.show(block=True)
"""
