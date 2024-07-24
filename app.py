"""
Stream lit GUI for hosting ANNCodeGen
"""

# Imports
import os
import json
import streamlit as st

from ANNCodeGen import *

# Main Vars
config = json.load(open("./StreamLitGUI/UIConfig.json", "r"))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    "Choose one of the following",
        tuple(
            [config["PROJECT_NAME"]] + 
            config["PROJECT_MODES"]
        )
    )
    
    if selected_box == config["PROJECT_NAME"]:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(" ", "_").lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config["PROJECT_NAME"])
    st.markdown("Github Repo: " + "[" + config["PROJECT_LINK"] + "](" + config["PROJECT_LINK"] + ")")
    st.markdown(config["PROJECT_DESC"])
    # st.write(open(config["PROJECT_README"], "r").read())

#############################################################################################################################
# Repo Based Vars
PATHS = {
    "cache": "StreamLitGUI/CacheData/Cache.json",
}

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    '''
    Load Cache
    '''
    global CACHE
    CACHE = json.load(open(PATHS["cache"], "r"))

def SaveCache():
    '''
    Save Cache
    '''
    global CACHE
    json.dump(CACHE, open(PATHS["cache"], "w"), indent=4)

# Main Functions


# UI Functions
def UI_GetInputs():
    '''
    UI - Get Inputs
    '''
    global classname
    global networkname
    global anntype
    global shortenedcode

    USERINPUT_InputCode = st.text_area(
        "Input Code", 
        value=DEFAULT_CODE, 
        height=300
    )
    USERINPUT_ClassName = st.text_input("Class Name")
    USERINPUT_NetworkName = st.text_input("Network Name")
    USERINPUT_ANNType = st.selectbox("ANN Type", ["nn"])
    # USERINPUT_ShortenedCode = st.checkbox("Shorten Code?")
    USERINPUT_ShortenedCode = False

    classname = USERINPUT_ClassName
    networkname = USERINPUT_NetworkName
    anntype = USERINPUT_ANNType
    shortenedcode = "Y" if USERINPUT_ShortenedCode else "N"

    USERINPUT_Inputs = {
        "code": USERINPUT_InputCode,
        "classname": classname,
        "networkname": networkname,
        "anntype": anntype,
        "shortenedcode": shortenedcode
    }
    return USERINPUT_Inputs

# Repo Based Functions
def generate_ann_code():
    # Title
    st.header("Generate ANN Code")

    # Prereq Loaders
    ResetStateVars()

    # Load Inputs
    USERINPUTS_Inputs = UI_GetInputs()

    # Process Inputs
    ANNCode = NetworkDescParser(
        code=USERINPUTS_Inputs["code"], 
        save=False, 
        param_classname=USERINPUTS_Inputs["classname"], 
        param_networkname=USERINPUTS_Inputs["networkname"], 
        param_anntype=USERINPUTS_Inputs["anntype"], 
        param_shortenedcode=USERINPUTS_Inputs["shortenedcode"]
    )

    # Display Outputs
    st.markdown("## ANN Code")
    st.markdown(f"```python\n{ANNCode}\n```")

#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()