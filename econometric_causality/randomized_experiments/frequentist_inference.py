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
import numpy as np

# User
from base.base_estimator import BaseCateEstimator

#------------------------------------------------------------------------------
# Empty class
#------------------------------------------------------------------------------
class CateEstimator(BaseCateEstimator):
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 verbose=False
                 ):
        # Initialize inputs
        self.verbose = verbose

    # --------------------
    # Class variables
    # --------------------

    # --------------------
    # Private functions
    # --------------------

    # --------------------
    # Public functions
    # --------------------
    def fit(self,Y,W):

        # Preprocess data
        super().preprocess_data(Y=Y,W=W)
        
        # Find mean and standard error for each treatment arm
        self.mean_outcome_per_treatment = {}
        self.var_outcome_per_treatment = {}
        for w in self.unique_treatments:
            self.mean_outcome_per_treatment[w]= Y[self.mask_treament[w]].mean()
            self.var_outcome_per_treatment[w] = Y[self.mask_treament[w]].var()
            
        return self
            
    def calculate_heterogeneous_treatment_effect(self):
        raise Exception(""""
                        Heterogeneous treatment effects are not available
                        for simple methods like sampling-based frequentist inference
                        """)
    
    def calculate_average_treatment_effect(self,w0=None,w1=None):
        
        if all([w is None for w in [w0,w1]]):
            w0 = self.unique_treatments[0]
            w1 = self.unique_treatments[1]
        
        # Compute the ATE as the difference in sample means of the outcome between group1 (treated) and group2 (control)
        tau = self.mean_outcome_per_treatment[w1]-self.mean_outcome_per_treatment[w0]
        tau_se = np.sqrt(self.var_outcome_per_treatment[w1]/self.n_obs[w1] + self.var_outcome_per_treatment[w0]/self.n_obs[w0])
        
        tau_obj = {"ate":tau,
                   "se":tau_se}
        
        return tau_obj

    def calculate_average_treatment_effect_via_regression(self):
        
        # TODO: Estimate regression of Y and W and let se be heteroskedasticity-robust standard error

        pass
        
        # tau_obj = {"ate":tau,
        #            "se":tau_se}
        
        # return tau_obj

