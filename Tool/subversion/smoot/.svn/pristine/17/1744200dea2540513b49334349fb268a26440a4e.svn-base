"""
The V (Verification) and V(Validation) for the models being used to predict the
outputs for the objective functions at the given set of validation input points. The 
emulator(prediction model) is built for each of the three functions individually.
Also, the simulator(the function equation) is used to calculate the true value at the 
set of inputs(validation data).

Author: Piyush Pandita

Date: 30 July 2015
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pydes
import design
import test_functions


__all__ = ['ModelCalibration']

class ModelCalibration(pydes.ParetoFront):
    """
    A sub-routine to verify and validate models built using GP surrogates(emulators)
    and to calibrate models with the optimal values for the parameters.
    """
    
    def models(self):
        """
        BUilding the emulators to predict the outputs at given inputs for validation.
        """
        
        self.model = self.emulator.build_gaussian_models()
        return self.model


    def standardized_prediction_error(self,models):
        """
        The standardized predicition error for each of the data validation points gives an idea of the conformation
        of the model predictions to the ideal experiment/simulator output.

        returns :   The standardized_prediction_error for each prediction for each model.
        """
        self.models = models
        self.predictions = self.emulator.make_predictions(models)
        standardized_prediction_error = []
        chi = []
        for j in xrange(self.Y_true.shape[0]):
            std_err = np.ndarray((self.Y_true.shape[1],1))
            chi_sq = 0
            for k in xrange(self.Y_true.shape[1]):
                std_err[k][0] = (self.Y_true[j][k][0] - self.predictions[j][0][k][0])/self.predictions[j][1][k,k]
                chi_sq = chi_sq + (std_err[k][0])**2
            standardized_prediction_error.append(std_err)
            chi.append(chi_sq)

        print 'chi squared distance is =',chi  #print standardized_prediction_error
        return standardized_prediction_error

    def mahalanobis_distance(self):
        """
        Computes the Mahalanobis Dsitance for eahc of the models/emulators that have been built.
        The MD is a statistic which has a F Snedecor distribution as the number
        of training points approaches infinity.

        :returns:   The Mahalanobis Distance ``MD`` for the given models, as a list.
        """
        self.mahalanobis=[]
        self.predictions = self.emulator.make_predictions(self.emulators)
        for j in xrange(self.Y_true.shape[0]):
            print self.predictions[j][1].shape
            g_mat = np.linalg.cholesky(self.predictions[j][1])
            g_inv = np.linalg.inv(g_mat)
            v_inv = np.linalg.inv(self.predictions[j][1])
            #print self.Y_true[j][:,0]
            #print self.predictions[j][0]
            pred_err = (self.Y_true[j][:,0] - self.predictions[j][0][:,0])[:,None]
            pred_err_t = pred_err.T
            d1 = np.dot(pred_err_t,v_inv)
            d2 = np.dot(d1,pred_err)
            #print pred_err.shape
            dot_mat = np.dot(g_inv,pred_err)
            dot_mat_t = dot_mat.T
            md = np.dot(dot_mat_t,dot_mat)
            self.mahalanobis.append(md)
        return self.mahalanobis

    def error_plots(self,ax,standardized_prediction_error):
        """
        Plots the standardized_prediction_error for each of the models wrt the the training set inputs.
        """
        for k in xrange(self.Y_true.shape[0]):
            for j in xrange(self.X_training.shape[1]):
                #print self.X_training.shape
                ax.scatter(self.X_validation[:,j],standardized_prediction_error[k][:,0],s=40)
                ax.set_xlabel('The input' + ' ' + str(j))
                ax.set_ylabel('The standardized error')
                plt.show(block=True)
                plt.clf()
                fig = plt.figure()
                ax = fig.add_subplot(111)


    def __init__(self,X_init,Y_init,X_design,Y_ref,obj_funcs):
        
        pydes.ParetoFront.__init__(self, X_init,Y_init,X_design,Y_ref,obj_funcs)
        
        self.X_training = X_init
        self.Y_training = Y_init
        self.X_validation = X_design
        self.Y_ref= Y_ref
        self.obj_funcs = obj_funcs
        self.Y_true = np.array([[[f(x)] for x in self.X_validation] for f in obj_funcs])
        self.emulator = pydes.ParetoFront(self.X_training,self.Y_training,self.X_validation,self.Y_ref,self.obj_funcs)
        self.emulators = self.models()
        #self.md = self.mahalanobis_distance()
        
    def __str__(self):
        return 'The Mahalanobis Distance and the Prediction errors are used as diagnostics'



"""
quit()

function_name='vlmop3'
obj_funcs, X_init, X_design, X_true = test_functions.select_function(function_name)

#X_init = design.latin_center(10,2)
X_validation = design.latin_center(1000,2)
X_init = design.latin_center(10,2)
#X_init = np.array([[0.15,0.65],[0.2,0.2],[0.81,0.3],[0.55,0.65],[0.65,0.73],[0.8,0.9],[0.85,0.9]])
#X_design = np.linspace(0,1,100)[:,None]
#X_design =  np.array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.55,0.65],[0.65,0.73],[0.8,0.9]])
#X_validation = np.array([[0.22,0.32],[0.3,0.4],[0.05,0.1],[0.5,0.55],[0.6,0.62],[0.7,0.75],[0.77,0.82],[0.35,0.45],[0.88,0.92],[0.92,0.95]])

#a = np.array([6*np.sin((np.pi)/12),(-2*(np.pi)*(np.sin((np.pi)/12)))])
#b = np.array([6*np.sin((np.pi)/12)+ (2*(np.pi)*(np.cos((np.pi)/12))), (6*(np.cos(np.pi/12)))])

#a = np.array([-np.pi,-5,-5])
#b = np.array([np.pi,5,5])
#a = np.array([-3,-3])
#b = np.array([3,3])
#X_init = X_init*(b-a) + a
#X_validation = X_design*(b-a) + a

Y_init = np.array([[[f(x)] for x in X_init] for f in obj_funcs])
Y_ref = 100*np.ones((3,1))
Y_true = np.array([[[f(x)] for x in X_validation] for f in obj_funcs])
models = []
model_validation = pydes.ParetoFront(X_init,Y_init,X_validation,Y_ref,obj_funcs)
models = model_validation.build_gaussian_models()
predictions = model_validation.make_predictions(models)

g = []
dm_original = []
for i in xrange(len(predictions)):
    jitter = 1e-2*np.eye(predictions[i][1].shape[0],predictions[i][1].shape[0])
    predictions1 = predictions[i][1]  + jitter
    d = np.linalg.eigh(predictions1)
       
    #print predictions[i][1]
    gg = np.linalg.cholesky(predictions1)
    v_inv = np.linalg.inv(predictions[i][1])
    gg_inv = np.linalg.inv(gg)
    yy = Y_true[i,:,0] - predictions[i][0][:,0]
    
    xx = np.dot(gg,yy)
    xx_t = xx.T
    a1 = np.dot(yy.T,v_inv)
    a2 = np.dot(a1,yy)
    dm_original.append(a2)
    dm = np.dot(xx_t,xx)
    g.append(dm)
       
print 'mahalanobis distance from equation 10 for the' + ' ' + str(Y_true.shape[0]) + 'models = ',dm_original
print 'mahalanobis distance by decomposing the covariance matrix for the'+ ' ' + str(Y_true.shape[0]) + ' models = ',g

# The chi squared diagnostics for the validation data

chi1 = []
chi2 = []
chi3 = []

for i in xrange(Y_true.shape[1]):
    #print predictions[0][1][i,i],predictions[1][1][i,i],predictions[2][1][i,i]
    chi1.append((Y_true[0,i,0] - predictions[0][0][i,0])/np.sqrt(predictions[0][1][i,i]))
    chi2.append((Y_true[1,i,0] - predictions[1][0][i,0])/np.sqrt(predictions[1][1][i,i]))
    chi3.append((Y_true[2,i,0] - predictions[2][0][i,0])/np.sqrt(predictions[2][1][i,i]))

chi = (chi1,chi2,chi3) # A tuple that holds the values of the standardized prediction errors for all the errors
x1=0
x2=0
x3=0

for i in xrange(len(chi1)):
    x1 = x1+(chi1[i])**2
    x2 = x2+(chi2[i])**2
    x3 = x3+(chi3[i])**2

print 'the chi squared values are = ',x1,x2,x3
quit()
fig = plt.figure()
ax = Axes3D(fig)
#ax.scatter(Y_init[0,:,0],Y_init[1,:,0],s=80,color='green')
ax.scatter(predictions[0][0][:,0],predictions[1][0][:,0],predictions[2][0][:,0],color='blue')
ax.scatter(Y_true[0,:,0],Y_true[1,:,0],Y_true[2,:,0],color='black')
plt.show(block=True)

for i in xrange(len(chi)):
    for j in xrange((X_validation.shape[1])):
        fig.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(X_validation[:,j],chi[i])
	ax.set_xlabel('The input '+' '+ str(j))
	ax.set_ylabel('The standardized prediction error')
	ax.set_title('The input' + str(j)+' against the standardized prediction error for the'+ str(i)+'th model')
	plt.savefig('standardized_prediction_for_the_input_'+str(j)+'_.png')
	plt.show(block=True)
    fig.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(predictions[i][0][:,0],chi[i])
    ax.set_xlabel('The prediction made by the GP emulator')
    ax.set_ylabel('The standardized prediction error')
    ax.set_title('Error vs the Prediction for the' +' ' + str(i)+ ' th model')
    plt.savefig('error_vs_prediction_'+str(i)+'_.png')
    plt.show(block=True)

"""
