# installing conda envs

given a yml file with python pacakges, one could simply do `mamba env create -f env.yml`. However, if you don't have write permissions to the default path, one could do: \
1. `conda create -p /path/to/directory/to/save/env/with/write/permission`\
2. `conda activate /path/to/directory/to/save/env/with/write/permission`\
3. `mamba env update -f env.yml` (note that name in the env.yml must be delete or commented out)