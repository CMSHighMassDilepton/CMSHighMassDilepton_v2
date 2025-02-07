## Documentation
Stage1 is first out of three stages: stage1, stage2, stage3

Stage1 reads data and MC NanoAOD samples and applies skim selection and saves the resulting output as a dictionary of akward arrays. The dictionary is saved as sets of parquet files in the user specified directory.
The keys that the output dictionary contains is defined in run_stage1.py as "skim_dict" variable.
