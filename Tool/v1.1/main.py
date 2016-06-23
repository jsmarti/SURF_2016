# ----------------------------------------------------------------------
#  MAIN PROGRAM - generated by the Rappture Builder
# ----------------------------------------------------------------------
import Rappture
import sys
import numpy as np
from test_object import Obj
import pickle as pkl
import base64
from ast import literal_eval

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


#########################################################
#  Add your code here for the main body of your program
#########################################################

try:
	f = open('model.obj','rb')
	model = pkl.load(f)
	f.close()
	model.add(int(new_result))
	x, y = model.get_data()
	f = open('model.obj','wb')
	pkl.dump(model,f,pkl.HIGHEST_PROTOCOL)
	f.close()
	new_design = 'Model loaded, data updated'
	if isinstance(inputs,tuple):
		new_design += ' , also, inputs are a tuple'
	
except IOError:
	model = Obj()
	model.add(int(new_result))
	x, y = model.get_data()
	f = open('model.obj','wb')
	pkl.dump(model,f,pkl.HIGHEST_PROTOCOL)
	f.close()
	new_design = 'Model created, data updated'
	if isinstance(literal_eval(inputs),tuple):
		new_design += ' , also, inputs are a tuple'
	
with open('test_pareto.png','rb') as img:
	imdata = base64.b64encode(img.read())

# spit out progress messages as you go along...
Rappture.Utils.progress(0, "Starting...")
Rappture.Utils.progress(5, "Loading data...")
Rappture.Utils.progress(50, "Half-way there")
Rappture.Utils.progress(100, "Done")

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
io['output.curve(curve).component.xy'] = (x, y)


io.close()
sys.exit()
