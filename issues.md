# Logical:

- [x] PH should be greater from lead, cmaps PH=2 is too small. (use the rule described in paper to decide)
- [x] Neirest Neighboard (see if output is always in [0,1]). Sometimes we get -inf and inf in scores.
      Solved after replacing log(0) with log(1) in the `estimate_for_subsequences` method in NP.
- [ ] self tunning is not compatible with techniques that return zero scores.

# Dependecies:

- [ ] When updating the enviroment: Pip subprocess error: ERROR: No matching distribution found for torch==1.9.0+cu111.
      The user nne to use
      the command : `pip install --force-reinstall torch==1.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/`

- [ ] Cython NotFoundModuleError (Need to install it before environment update)

- [ ] SAND needs sklearn <0.24 and tslearn==0.4.1 but profile based needs sklearn >=1, where sklearn Problem with building wheel for sklearn 0.23 (using pyproject.toml) (worked when installing 0.24).
      It may be handle if we dont use cross-correlaiton distance.
- [ ] Why c_maps folder not in DataFolder

# TO-DO

- [x] Add all dataset characteristics to utils.loadDataset.get_dataset, (like PH and column names etc.) , so the user define only parameters, at least to the dataset will be used for the analyses.
      Solved: Now there is a method in utils that returns a discretionary with all parameters needed for pipeline along whit data.
