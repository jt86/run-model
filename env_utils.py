import os

def get_working_dir(script_or_model):
    prefix_dict = {"GRIFF" : "C:\\Users\\alexg",
                  "PRED" : "D:",
                  "LAPTOP-280K62JU" : "C:\\Users\\bobby",
                  "DESKTOP-I60HQJL" : "C:\\Users\\Joe"}
    try:
        prefix = prefix_dict[os.environ['COMPUTERNAME']]
    except:
        return "Device not recognised"
    
    if script_or_model=='script':
        os.chdir(f"{prefix}\\PEP Health\\Tech - Tech\\Scripts\\Python")
    if script_or_model=='model':
        os.chdir(f"{prefix}\\PEP Health\\Tech - Tech\\Models\\Models")
get_working_dir('model')