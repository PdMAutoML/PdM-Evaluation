## Useful staff

### events df 
has columns: [date,type,description,source]

### run to failure
The data are considered as run to failure when no `'failure'` preferences are set or they are equal to `None`

### Mehtod info:

#### SAND: 
 sub-sequence=k*pattern_length
 overlaping==sub-sequence (default)
 init_batch>batch>sub-sequence

# TO DO:
- Add raw level dataset preprocessing as step on pipeline
- Add distance metrics from TSB to utils
- Add thresholders