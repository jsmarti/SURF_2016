# ----------------------------------------------------------------------
#  MAIN PROGRAM - generated by the Rappture Builder
# ----------------------------------------------------------------------
import Rappture
import sys
import numpy as np
import pickle as pkl
import dill
import base64
from ast import literal_eval
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import design
import matplotlib
matplotlib.use('PS')
from matplot import * # MATPLOT GOES BEFORE ANYTHING ELSE
from pydes import *
from mpl_toolkits.mplot3d import Axes3D
from math import *
import math
import csv
import copy
import shutil

# uncomment these to redirect stdout and stderr
# to files for debugging.
#sys.stderr = open('debug.err', 'w')
#sys.stdout = open('debug.out', 'w')

# open the XML file containing the run parameters
io = Rappture.PyXml(sys.argv[1])

#########################################################
# Get input values from Rappture
#########################################################


# get input value for input.phase(desc).note(note)
note = io['input.phase(desc).note(note).current'].value

# get input value for input.phase(initial_inputs).note(note_setp_2)
note_setp_2 = io['input.phase(initial_inputs).note(note_setp_2).current'].value

# get input value for input.phase(initial_inputs).string(inputs)
inputs = io['input.phase(initial_inputs).string(inputs).current'].value

# get input value for input.phase(initial_inputs).string(outputs)
outputs = io['input.phase(initial_inputs).string(outputs).current'].value

# get input value for input.phase(initial_inputs).string(l_bounds)
l_bounds = io['input.phase(initial_inputs).string(l_bounds).current'].value

# get input value for input.phase(initial_inputs).string(u_bounds)
u_bounds = io['input.phase(initial_inputs).string(u_bounds).current'].value

# get input value for input.phase(initial_inputs).integer(max_it)
max_it = int(io['input.phase(initial_inputs).integer(max_it).current'].value)

# get input value for input.phase(iterative_run).note(note_step_3)
note_step_3 = io['input.phase(iterative_run).note(note_step_3).current'].value

# get input value for input.phase(iterative_run).boolean(finish)
# returns value as string "yes" or "no"
finish = io['input.phase(iterative_run).boolean(finish).current'].value == 'yes'

# get input value for input.phase(iterative_run).string(new_result)
new_result = io['input.phase(iterative_run).string(new_result).current'].value


#Pareto Model
#pareto_model = None

#########################################################
#  Add your code here for the main body of your program
#########################################################

########################################################
# My methods
########################################################

def check_observations():
	'''
	Check for all the input observations to be empty or to have the correct tuple
	format
	'''
	global inputs, outputs, l_bounds, u_bounds, max_it

	try:
		check_inputs = inputs == '' or isinstance(literal_eval(inputs), tuple)
		check_outputs = outputs == '' or isinstance(literal_eval(outputs), tuple)
		check_l_bounds = l_bounds == '' or isinstance(literal_eval(l_bounds), tuple)
		check_u_bounds = u_bounds == '' or isinstance(literal_eval(u_bounds), tuple)
		check_max_it = max_it == '' or isinstance(max_it,int)

		check = check_inputs and check_outputs and check_l_bounds and check_u_bounds and check_max_it
		return check
	except:
		return False

def check_new_result():
	'''
	Checks that the new result has the correct tuple format
	'''
	global new_result
	try:
		check = isinstance(literal_eval(new_result), tuple)
		return check
	except:
		return False

def existing_model():
	'''Looks for a previous model'''
	return os.path.isfile('model.obj')

def restart():
	'''Restarts the program'''
	try:
		os.system('rm model.obj')
		os.system('rm -rf surf_test_results_noisy_moo/')
	except:
		pass

def new_optimization():
	global inputs, outputs, l_bounds, u_bounds, max_it
	response = None
	if check_observations():
		Rappture.Utils.progress(10, "New model being created...")
		out_dir = 'surf_test_results_noisy_moo'
		if os.path.isdir(out_dir):
			shutil.rmtree(out_dir)
		os.makedirs(out_dir)

		X_init = literal_eval(inputs)
		Y_init = literal_eval(outputs)
		X_init = np.array(X_init)
		Y_init = np.array(Y_init)

		a = literal_eval(l_bounds)
		b = literal_eval(u_bounds)
		a = np.array(a)
		b = np.array(b)
		X_design = (b-a)*design.latin_center(1000, 2, seed=314519) + a
		pareto_model = ParetoFront(X_init, Y_init, X_design=X_design, gp_opt_num_restarts=50, verbose=False, max_it=max_it, make_plots=True, add_at_least=30, get_fig=get_full_fig, fig_prefix=os.path.join(out_dir,'ex1'), Y_true_pareto=None, gp_fixed_noise=None, samp=100, denoised=True)
		Rappture.Utils.progress(20, "Performing optimization algorithm...")
		pareto_model.optimize_paused()
		response = pareto_model.get_response()
		Rappture.Utils.progress(60, "Saving the model...")
		model_file = open('model.obj','wb')
		pkl.dump(pareto_model, model_file, pkl.HIGHEST_PROTOCOL)
		model_file.close()
		Rappture.Utils.progress(100, "Done...")

	else:
		response = 'Incorrect tuples for new model'

	return response

def continue_optimization():
	global finish
	response = None
	if finish:
		restart()
		response = 'Program restarted, enter new input observations'
	elif check_new_result():
		Rappture.Utils.progress(10, "Loading previous model...")
		model_file = open('model.obj','rb')
		pareto_model = pkl.load(model_file)
		model_file.close()
		Rappture.Utils.progress(20, "Performing optimization algorithm...")
		pareto_model.optimize_paused(literal_eval(new_result))
		response = pareto_model.get_response()
		Rappture.Utils.progress(60, "Saving the model...")
		model_file = open('model.obj','wb')
		pkl.dump(pareto_model, model_file, pkl.HIGHEST_PROTOCOL)
		model_file.close()
		Rappture.Utils.progress(100, "Done...")
	else:
		response = 'Incorrect tuple for new result'

	return response

#Check for a previous state of the program

Rappture.Utils.progress(0, "Starting...")
if not existing_model():
	new_design = new_optimization()
else:
	new_design = continue_optimization()

#Pareto front
path = 'surf_test_results_noisy_moo/'
try:
	fronts = [front for front in os.listdir(path) if front.endswith('.png')]
	if len(fronts) == 0:
		image_path = 'no_pareto.png'
	else:
		fronts.sort()
		last_front = fronts[len(fronts) - 1]
		image_path = path + last_front
except:
	image_path = 'no_pareto.png'

with open(image_path,'rb') as img:
	imdata = base64.b64encode(img.read())

#########################################################
# Save output values back to Rappture
#########################################################

# save output value for output.string(new_design)
io['output.string(new_design).current'] = new_design

# save output value for output.image(pareto_front)
# data should be base64-encoded image data
io['output.image(pareto_front).current'] = imdata

# save output value for output.curve(curve)
# x and y should be lists or arrays -- modify as needed
x = np.linspace(0,1,1000)
y = np.sin(2*np.pi*1000*x)
io['output.curve(curve).component.xy'] = (x, y)


io.close()
sys.exit()
