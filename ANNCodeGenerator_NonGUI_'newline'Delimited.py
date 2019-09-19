#--Imports------------------------------------------------------------------
import re
import random
#---------------------------------------------------------------------------

#--Initialisations----------------------------------------------------------
first_input = True

fields = 'ANN Description Location (with filename)', 'ANN Class Name', 'ANN Network Name', 'Destination Folder', 'ANN Type', 'Shortened Code'

NetworkParameters = 'loss_fn', 'optimizer'

inputfileloc = ''
classname = ''
destfolder = ''

networkname = ''

anntype = ''	# Keras or NN module

shortenedcode = False 	# Whether to use loops in output ann code or put every layer as a separate line

network_desc = []

n_inputs_name = 'n_inputs'
n_outputs_name = 'n_outputs'

constructor_params = [n_inputs_name, n_outputs_name]

networkcode_format = ['^nng_imports^', 
						'class ^nng_classname^(^nng_classparams^):', 
							'', 
							'\tdef __init__(self ^nng_constructorparams^):', 
								'\t\t^nng_constructorimports^', 
								'\t\tsuper().__init__()', 
								'\t\ttorch.manual_seed(0)', 
								'\t\tseq = []', 
								'^nng_buildnetwork^', 
								'\t\tself.^nng_networkname^ = nn.Sequential(', 
								'\t\tOrderedDict(seq)', 
								'\t\t)', 
							'', 
							'\tdef forward(self, X):', 
								'\t\treturn self.^nng_networkname^(X)', 
							'', 
							'\tdef fit(self, x, y, opt, loss_fn, epochs, display_loss=True):', 
								'\t\tfrom torch import optim', 
								'\t\timport matplotlib.pyplot as plt', 
								'\t\timport matplotlib.colors', 
								'', 
								'\t\tloss_arr = []', 
								'\t\tfor epoch in range(epochs):', 
									'\t\t\tloss = self.loss_fn(self.forward(x), y)', 
									'\t\t\tloss_temp = loss.item()', 
									'\t\t\tloss_arr.append(loss_temp)', 
									'', 
									'\t\t\tloss.backward()', 
									'\t\t\topt.step()', 
									'\t\t\topt.zero_grad()', 
									'', 
								'\t\tif display_loss:', 
									'\t\t\tplt.plot(loss_arr)', 
									'\t\t\tplt.xlabel(\'Epochs\')', 
									'\t\t\tplt.ylabel(\'CE\')', 
									'\t\t\tplt.show()', 
									'', 
								'\t\treturn loss.item()', 
								'', 
							'\tdef predict(self, X):', 
								'\t\timport numpy as np', 
								'\t\tY_pred = self.net(X)', 
								'\t\tY_pred = Y_pred.detach().numpy()', 
								'\t\treturn np.array(Y_pred).squeeze()'
						]

format_values = [['^nng_imports^', ''], ['^nng_constructorimports^', ''], ['^nng_classname^', ''], ['^nng_networkname^', ''], ['^nng_classparams^', ''], ['^nng_constructorparams^', ''], ['^nng_buildnetwork^', '']]

var_values = [['$var(n_inputs)$', ''], ['$var(n_inputs)$', ''], ['$var(n_inputs)$', '']]

keras_constructor_imports = [
							"import keras", "from keras.models import Sequential", 
							"from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D", 
							"from keras.optimizers import SGD", 
							"from keras.layers.advanced_activations import LeakyReLU"
							]
nn_constructor_imports = ["import torch", "import torch.nn as nn", "from collections import OrderedDict"]

#---------------------------------------------------------------------------

#--Functions----------------------------------------------------------------
#--1--
def fetch_inputs(fields):
	global first_input
	input_dict = {}

	for field in fields:
		text  = input(field + ": ")
		if field == "EnterArrayTypeDataFieldHere":
			text = text.split(",")
		input_dict[field] = text
	print(input_dict)

	first_input = False

	global inputfileloc
	global classname
	global destfolder
	global networkname
	global anntype
	global shortenedcode

	if input_dict["ANN Description Location (with filename)"] != '':
		inputfileloc = input_dict["ANN Description Location (with filename)"]
	if input_dict["ANN Class Name"] != '':
		classname = input_dict["ANN Class Name"]
	if input_dict["Destination Folder"] != '':
		destfolder = input_dict["Destination Folder"]
	if input_dict["ANN Network Name"] != '':
		networkname = input_dict["ANN Network Name"]
	if input_dict["ANN Type"] != '':
		anntype = input_dict["ANN Type"]
	if input_dict["Shortened Code"] != '':
		shortenedcode = input_dict["Shortened Code"]

	NetworkDescParser(filepath=inputfileloc, destfolder=destfolder)

#--2--
def FileContents(filepath):
	return open(filepath, "r").read()

#--3--
def NetworkDescParser(filepath, destfolder):
	global anntype

	global network_desc

	contents = FileContents(filepath)
	contents = contents.split("\n")

	print("\n\nCONTENTS: ", contents, "\n\n")

	last_linear_layer_index = 0
	network_index = 0

	network_started = False

	for line in contents:
		print ("Line(wos): ", line)
		line = line.strip()	
		print ("Line(ws): ", line)

		if re.search('StartNetworkDesc', line):
			network_started = True
			continue

		if re.search('EndNetworkDesc', line):
			network_started = False

		###################################### -- FORMAT REQUIRED TO CHECK -- ###########################################
		# FORMAT -- Linear<2><3> - Order predefined   ^[^<>]
		

		if network_started:			
			print("params: ", re.findall('<([^<>]*)>', line))
			print("name: ", re.findall('^([^<>]*)<', line))
			params = re.findall('<([^<>]*)>', line)
			network_desc.append([re.findall('^([^<>]*)<', line)[0].strip(), params])

			if params[0] != '' and params[0] != ' ':
				last_linear_layer_index = network_index
			network_index += 1

		


	###################################### -- FORMAT REQUIRED TO CHECK -- ###########################################

	print(network_desc)

	# First Layer must have n_inputs as input size
	first_layer = network_desc[0]
	first_layer_params = first_layer[1]
	first_layer_params[0] = n_inputs_name
	first_layer[1] = first_layer_params
	network_desc[0] = first_layer

	# Last Layer must have n_outputs as output size
	last_layer = network_desc[last_linear_layer_index]
	last_layer_params = last_layer[1]
	last_layer_params[1] = n_outputs_name
	last_layer[1] = last_layer_params
	network_desc[last_linear_layer_index] = last_layer

	AssignFormatValues()

	CreateOutputFile()

#--4--
def AssignFormatValues():
	global keras_constructor_imports
	global anntype
	global nn_constructor_imports
	global classname
	global networkname
	global shortenedcode
	global constructor_params

	for f in format_values:
		if f[0] == '^nng_constructorimports^':
			if anntype == 'keras':
				for imp in keras_constructor_imports:
					f[1] = f[1] + imp + "\n\t\t"
			elif anntype == 'nn':
				for imp in nn_constructor_imports:
					f[1] = f[1] + imp + "\n\t\t"

		if f[0] == '^nng_classname^':
			if classname != '':
				f[1] = classname
			else:
				 f[1] = "NeuralNetwork"

		if f[0] == '^nng_networkname^':
			if networkname != '':
				f[1] = networkname
			else:
				 f[1] = "net"

		if f[0] == '^nng_classparams^':
			if anntype == 'nn':
				f[1] = 'nn.Module'

		if f[0] == '^nng_buildnetwork^':
			if shortenedcode in ['Y', 'Yes', 'yes', 'y', '1']:
				f[1] = BuildNetworkCode(True)
			else:
				f[1] = BuildNetworkCode(False)

		if f[0] == '^nng_constructorparams^':
			for p in constructor_params:
				f[1] = f[1] + ', ' + p

	print("FORMAT: ", format_values)

#--5--
def BuildNetworkCode(shortenedcode, tabspace='\t\t'):
	global network_desc
	global anntype

	network_text = ''
	if anntype == 'nn':
		if not shortenedcode:
			for layer in network_desc:
				params = ''
				for param in layer[1]:
					params += ', ' + param.strip()
				params = params[params.find(',')+1:]

				network_text += tabspace + 'nn.' + layer[0].strip() + '(' + params.strip() + ')' + '\n'
			return network_text

#--6--
def CreateOutputFile():
	global networkcode_format
	global format_values

	global destfolder

	contents = ''
	for c in networkcode_format:

		for fv in format_values:
			c = c.replace(fv[0], fv[1])
		contents = contents + c
		contents = contents + '\n'
	print("OutputFileContents: ", contents)

	f = open(destfolder + '\\' + classname + '_Code.py', 'w+')
	f.write(contents)


#---------------------------------------------------------------------------

#--Main Code----------------------------------------------------------------

fetch_inputs(fields)



Linear_params = ['n_inputs', 'n_outputs']
Sigmoid_params = []
Softmax_params = []