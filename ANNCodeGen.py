"""
ANN Code Generator
"""

# Imports
import re
import random

# Main Vars
VERBOSE = False
first_input = True
fields = "ANN Description Location (with filename)", "ANN Class Name", "ANN Network Name", "Destination Folder", "ANN Type", "Shortened Code"
NetworkParameters = "loss_fn", "optimizer"
inputfileloc = ""
classname = ""
destfolder = ""
networkname = ""
## Keras or NN module
anntype = ""    
## Whether to use loops in output ann code or put every layer as a separate line
shortenedcode = False

# State Vars
network_desc = None
n_inputs_name = None
n_outputs_name = None
constructor_params = None
networkcode_format = None
format_values = None
var_values = None
keras_constructor_imports = None
nn_constructor_imports = None

# Main Functions
def ResetStateVars():
    '''
    Reset State Vars
    '''
    global network_desc
    global n_inputs_name
    global n_outputs_name
    global constructor_params
    global networkcode_format
    global format_values
    global var_values
    global keras_constructor_imports
    global nn_constructor_imports

    network_desc = []
    n_inputs_name = "n_inputs"
    n_outputs_name = "n_outputs"
    constructor_params = [n_inputs_name, n_outputs_name]
    networkcode_format = [
        "^nng_imports^", 
        "class ^nng_classname^(^nng_classparams^):", 
            "", 
            "\tdef __init__(self ^nng_constructorparams^):", 
                "\t\t^nng_constructorimports^", 
                "\t\tsuper().__init__()", 
                "\t\ttorch.manual_seed(0)", 
                "", 
                "\t\tself.^nng_networkname^ = nn.Sequential(", 
                "^nng_buildnetwork^", 
                "\t\t)", 
            "", 
            "\tdef forward(self, X):", 
                "\t\treturn self.^nng_networkname^(X)", 
            "", 
            "\tdef fit(self, x, y, opt, loss_fn, epochs, display_loss=True):", 
                "\t\tfrom torch import optim", 
                "\t\timport matplotlib.pyplot as plt", 
                "\t\timport matplotlib.colors", 
                "", 
                "\t\tloss_arr = []", 
                "\t\tfor epoch in range(epochs):", 
                    "\t\t\tloss = self.loss_fn(self.forward(x), y)", 
                    "\t\t\tloss_temp = loss.item()", 
                    "\t\t\tloss_arr.append(loss_temp)", 
                    "", 
                    "\t\t\tloss.backward()", 
                    "\t\t\topt.step()", 
                    "\t\t\topt.zero_grad()", 
                    "", 
                "\t\tif display_loss:", 
                    "\t\t\tplt.plot(loss_arr)", 
                    "\t\t\tplt.xlabel(\"Epochs\")", 
                    "\t\t\tplt.ylabel(\"CE\")", 
                    "\t\t\tplt.show()", 
                    "", 
                "\t\treturn loss.item()", 
                "", 
            "\tdef predict(self, X):", 
                "\t\timport numpy as np", 
                "\t\tY_pred = self.net(X)", 
                "\t\tY_pred = Y_pred.detach().numpy()", 
                "\t\treturn np.array(Y_pred).squeeze()"
    ]
    format_values = [
        ["^nng_imports^", ""], 
        ["^nng_constructorimports^", ""], 
        ["^nng_classname^", ""], 
        ["^nng_networkname^", ""], 
        ["^nng_classparams^", ""], 
        ["^nng_constructorparams^", ""], 
        ["^nng_buildnetwork^", ""]
    ]
    var_values = [
        ["$var(n_inputs)$", ""], 
        ["$var(n_inputs)$", ""], 
        ["$var(n_inputs)$", ""]
    ]
    keras_constructor_imports = [
        "import keras", "from keras.models import Sequential", 
        "from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D", 
        "from keras.optimizers import SGD", 
        "from keras.layers.advanced_activations import LeakyReLU"
    ]
    nn_constructor_imports = ["import torch", "import torch.nn as nn", "from collections import OrderedDict"]

def fetch_inputs(fields):
    '''
    Fetch Inputs
    '''
    global first_input
    global inputfileloc
    global classname
    global destfolder
    global networkname
    global anntype
    global shortenedcode

    input_dict = {}
    for field in fields:
        text  = input(field + ": ")
        if field == "EnterArrayTypeDataFieldHere":
            text = text.split(",")
        input_dict[field] = text
    if VERBOSE: print("\n\n ### input_dict:\n", input_dict, "\n\n")
    first_input = False

    if input_dict["ANN Description Location (with filename)"] != "":
        inputfileloc = input_dict["ANN Description Location (with filename)"]
    if input_dict["ANN Class Name"] != "":
        classname = input_dict["ANN Class Name"]
    if input_dict["Destination Folder"] != "":
        destfolder = input_dict["Destination Folder"]
    if input_dict["ANN Network Name"] != "":
        networkname = input_dict["ANN Network Name"]
    if input_dict["ANN Type"] != "":
        anntype = input_dict["ANN Type"]
    if input_dict["Shortened Code"] != "":
        shortenedcode = input_dict["Shortened Code"]

    NetworkDescParser(filepath=inputfileloc)

def FileContents(filepath):
    '''
    File Contents
    '''
    return open(filepath, "r").read()

def NetworkDescParser(
        filepath=None, code=None, 
        save=True, 
        param_classname=None, param_networkname=None, param_anntype=None, param_shortenedcode=None
):
    '''
    Network Desc Parser
    '''
    global classname
    global networkname
    global anntype
    global shortenedcode
    global network_desc

    if param_classname is not None: classname = param_classname
    if param_networkname is not None: networkname = param_networkname
    if param_anntype is not None: anntype = param_anntype
    if param_shortenedcode is not None: shortenedcode = param_shortenedcode

    contents = []
    if filepath is not None:
        contents = FileContents(filepath)
        contents = contents.split("\n")
    elif code is not None:
        contents = code.split("\n")
    if VERBOSE: print("\n\n ### CONTENTS: ", contents, "\n\n")

    last_linear_layer_index = 0
    network_index = 0

    network_started = False
    for line in contents:
        if VERBOSE: print ("Line(wos): ", line)
        line = line.strip()    
        if VERBOSE: print ("Line(ws): ", line)

        if re.search("StartNetworkDesc", line):
            network_started = True
            continue

        if re.search("EndNetworkDesc", line):
            network_started = False

        # FORMAT -- Linear<2><3> - Order predefined   ^[^<>]
        if network_started:            
            if VERBOSE: print("params: ", re.findall("<([^<>]*)>", line))
            if VERBOSE: print("name: ", re.findall("^([^<>]*)<", line))
            params = re.findall("<([^<>]*)>", line)
            network_desc.append([re.findall("^([^<>]*)<", line)[0].strip(), params])

            if params[0] != "" and params[0] != " ":
                last_linear_layer_index = network_index
            network_index += 1
    if VERBOSE: print("\n\n ### network_desc:\n", network_desc, "\n\n")

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
    return CreateOutputFile(save=save)

def AssignFormatValues():
    '''
    Assign Format Values
    '''
    global keras_constructor_imports
    global anntype
    global nn_constructor_imports
    global classname
    global networkname
    global shortenedcode
    global constructor_params

    for f in format_values:
        if f[0] == "^nng_constructorimports^":
            if anntype == "keras":
                for imp in keras_constructor_imports:
                    f[1] = f[1] + imp + "\n\t\t"
            elif anntype == "nn":
                for imp in nn_constructor_imports:
                    f[1] = f[1] + imp + "\n\t\t"

        if f[0] == "^nng_classname^":
            if classname != "":
                f[1] = classname
            else:
                 f[1] = "NeuralNetwork"

        if f[0] == "^nng_networkname^":
            if networkname != "":
                f[1] = networkname
            else:
                 f[1] = "net"

        if f[0] == "^nng_classparams^":
            if anntype == "nn":
                f[1] = "nn.Module"

        if f[0] == "^nng_buildnetwork^":
            if shortenedcode in ["Y", "Yes", "yes", "y", "1"]:
                f[1] = BuildNetworkCode(True)
            else:
                f[1] = BuildNetworkCode(False)

        if f[0] == "^nng_constructorparams^":
            for p in constructor_params:
                f[1] = f[1] + ", " + p

    if VERBOSE: print("\n\n ### FORMAT: ", format_values, "\n\n")

def BuildNetworkCode(shortenedcode, tabspace="\t\t"):
    '''
    Build Network Code
    '''
    global network_desc
    global anntype

    network_text = ""
    if anntype == "nn":
        if not shortenedcode:
            for layer in network_desc:
                params = ""
                for param in layer[1]:
                    params += ", " + param.strip()
                params = params[params.find(",")+1:]
                network_text += tabspace + "nn." + layer[0].strip() + "(" + params.strip() + "), " + "\n"
            network_text = network_text[:network_text.rfind(",")] + " " + network_text[network_text.rfind(",")+1]
    return network_text

def CreateOutputFile(save=True):
    '''
    Create Output File
    '''
    global networkcode_format
    global format_values
    global destfolder

    contents = ""
    for c in networkcode_format:
        for fv in format_values:
            c = c.replace(fv[0], fv[1])
        contents = contents + c
        contents = contents + "\n"
    if VERBOSE: print("\n\n ### OutputFileContents: ", contents, "\n\n")

    if save:
        f = open(destfolder + "\\" + classname + "_Code.py", "w+")
        f.write(contents)

    return contents

# RunCode
ResetStateVars()
# fetch_inputs(fields)
Linear_params = ["n_inputs", "n_outputs"]
Sigmoid_params = []
Softmax_params = []

DEFAULT_CODE = """StartNetworkDesc
Linear <-><8>
Sigmoid <>
Linear <8><16>
Sigmoid <>
Linear <16><->
Softmax <>
EndNetworkDesc"""