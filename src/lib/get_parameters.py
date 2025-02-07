import glob
from omegaconf import OmegaConf

def getParametersForYr(parameter_path: str, year: str) -> dict:
    """
    This is a simple python function that takes in all the parameters defined by the local yaml files, merges them and returns a dictionary of omegaconf variables (which are basically dictionaries) for a given year
    If you would like to only accept certain yaml files, feel free to hard code the 
    filelist varaibles to contain the yaml files you want

    Params:
    parameter_path -> path where parameter yaml files are saved in 
        typically, the value is configs/parameters/
    year -> Run era year in question
    """
    filelist = glob.glob(parameter_path + "*.yaml")
    # print(f"getParametersForYr filelist: {filelist}")
    params = [OmegaConf.load(f) for f in filelist]
    merged_param = OmegaConf.merge(*params)
    yr_specific_params = {}
    for key, val in merged_param.items():
        if "cross_sections" in key:
            yr_specific_params[key] = val
        elif "jec" in key: # if jec, then do it separately
            sub_jec_pars = {}
            for sub_key, sub_val in val.items():
                sub_jec_pars[sub_key] = sub_val[year]
            yr_specific_params[key] = sub_jec_pars
        else:
            yr_specific_params[key] = val[year]
    yr_specific_params["do_roccor"] = True
    yr_specific_params["do_fsr"] = True
    yr_specific_params["do_geofit"] = True
    yr_specific_params["year"] = year
    yr_specific_params["do_jecunc"] = False
    yr_specific_params["do_jerunc"] = False

    # save year specific yaml for testing vs json version
    # directory = "./config"
    # filename = directory+"/parameters.yaml"
    # with open(filename, "w") as file:
    #     OmegaConf.save(config=yr_specific_params, f=file.name)
    return yr_specific_params