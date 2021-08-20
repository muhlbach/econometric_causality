#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
import more_itertools

# User
from base.base_estimator import BaseCateEstimator
from randomized_experiments import frequentist_inference
from utils.exceptions import CateError

#------------------------------------------------------------------------------
# Treatment Effect Estimator
#------------------------------------------------------------------------------
class TreatmentEffectEstimator(BaseCateEstimator):
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
        
        if len(W)>50:        
            raise Exception(
                f"""
                Randomization inference considers all possible treatment assignments.
                If we have more than 50-100 units, this will be computationally heavy.
                Current number of units: {len(W)}
                """)
        
        # Instantiate
        ate_estimator = frequentist_inference.TreatmentEffectEstimator(use_regression=False)
        
        # Compute tau hat
        ate_estimator.fit(Y=Y,W=W)
        self.tau_hat = ate_estimator.calculate_average_treatment_effect()
        
        self.tau_w = []
        for W_aug in more_itertools.distinct_permutations(np.array(W)):
            # Convert to series
            W_aug = pd.Series(W_aug)
            
            difference_estimator = frequentist_inference.TreatmentEffectEstimator(use_regression=False)
            
            difference_estimator.fit(Y=Y,W=W_aug)
            
            self.tau_w.append(difference_estimator.calculate_average_treatment_effect()['ate'])
            
        return self
    
    def calculate_heterogeneous_treatment_effect(self):
        raise CateError
    
    def calculate_average_treatment_effect(self):
        """
        """
        tau_abs = abs(self.tau_hat['ate'])
        tau_pvalue = np.mean([abs(tau_hat) >= tau_abs for tau_hat in self.tau_w])
        
        tau_obj = {
            "ate":self.tau_hat['ate'],
            "se":None,
            "p_value":tau_pvalue
            }
        
        tau_obj = {**self.tau_hat,**tau_obj}
        
        return tau_obj
    