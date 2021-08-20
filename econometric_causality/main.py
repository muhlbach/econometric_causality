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
# Standard
from sklearn.ensemble import RandomForestRegressor

from utils import data
from randomized_experiments import frequentist_inference
from randomized_experiments import randomization_inference
from selection_on_observables import (regression_adjustment, matching,
                                      inverse_probability_weighting)

#------------------------------------------------------------------------------
# Randomized Experiments
#------------------------------------------------------------------------------
# Generate data
Y, W, X = data.generate_data_rct(N=100,p=2)

#-----------------------------------
# Frequentist inference
#-----------------------------------
# Instantiate
frequentist_estimator = frequentist_inference.TreatmentEffectEstimator(use_regression=False)

# Fit
frequentist_estimator.fit(Y=Y,W=W)

# Estimate ATE
frequentist_ate = frequentist_estimator.calculate_average_treatment_effect(w0=0,w1=1)


#-----------------------------------
# Randomization inference
#-----------------------------------
# Instantiate
randomization_estimator = randomization_inference.TreatmentEffectEstimator()

# Fit
randomization_estimator.fit(Y=Y,W=W)

# Estimate ATE
randomization_ate = randomization_estimator.calculate_average_treatment_effect()


#------------------------------------------------------------------------------
# Selection on observables
#------------------------------------------------------------------------------
#-----------------------------------
# Regression adjustment
#-----------------------------------
# Instantiate
reg_adj_estimator = regression_adjustment.TreatmentEffectEstimator()

# Fit
reg_adj_estimator.fit(Y=Y,W=W,X=X)

# Estimate ATE
reg_adj_ate = reg_adj_estimator.calculate_average_treatment_effect()


#-----------------------------------
# Matching on covariates
#-----------------------------------
# Instantiate
matching_X_estimator = matching.MatchingOnCovariates()

# Fit
matching_X_estimator.fit(Y=Y,W=W,X=X)

# Estimate ATE
matching_X_ate = matching_X_estimator.calculate_average_treatment_effect()


#-----------------------------------
# Matching on propensity score
#-----------------------------------
# Instantiate
matching_ps_estimator = matching.MatchingOnPropensityScore()

# Fit
matching_ps_estimator.fit(Y=Y,W=W,X=X)

# Estimate ATE
matching_ps_ate = matching_ps_estimator.calculate_average_treatment_effect()


#-----------------------------------
# Inverse probability weighting
#-----------------------------------
# Instantiate
ipw_estimator = inverse_probability_weighting.TreatmentEffectEstimator()

# Fit
ipw_estimator.fit(Y=Y,W=W,X=X,propensity_score_estimator=RandomForestRegressor())

# Estimate ATE
ipw_ate = ipw_estimator.calculate_average_treatment_effect()














