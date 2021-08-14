#------------------------------------------------------------------------------
# Debugging
#------------------------------------------------------------------------------
# import os
# # Manually set path of current file
# path_to_here = "/Users/muhlbach/Repositories/econometric_causality/econometric_causality/"
# # Change path
# os.chdir(path_to_here)
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
from utils import data
from randomized_experiments import frequentist_inference

#------------------------------------------------------------------------------
# Randomized Experiments
#------------------------------------------------------------------------------
# Generate data
Y, W, X = data.generate_data_rct(N=1000,p=5)


cateestimator = frequentist_inference.CateEstimator()
self=cateestimator

cateestimator.fit(Y=Y,W=W)

cateestimator.calculate_average_treatment_effect()


cateestimator.preproc
cateestimator.mask_D0