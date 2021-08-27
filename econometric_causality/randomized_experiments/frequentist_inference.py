#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np

# User
from base.base_estimator import BaseCateEstimator
from regression.linear_regression import OLS
from utils.exceptions import CateError

#------------------------------------------------------------------------------
# Treatment Effect Estimator
#------------------------------------------------------------------------------
# TODO: Implement small sample adjustments according to Imbens & Kolesár (2016)
class TreatmentEffectEstimator(BaseCateEstimator):
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 use_regression=False,
                 verbose=False
                 ):
        # Initialize inputs
        self.use_regression=use_regression
        self.verbose = verbose

    # --------------------
    # Class variables
    # --------------------

    # --------------------
    # Private functions
    # --------------------
    def _calculate_average_treatment_effect_via_differences(self,w0=None,w1=None):
        
        if all([w is None for w in [w0,w1]]):
            w0 = self.unique_treatments[0]
            w1 = self.unique_treatments[1]
        
        # Compute the ATE as the difference in sample means of the outcome between group1 (treated) and group2 (control)
        tau = self.mean_outcome_per_treatment[w1] - self.mean_outcome_per_treatment[w0]
        tau_se = np.sqrt(self.var_outcome_per_treatment[w1]/self.n_obs[w1] + self.var_outcome_per_treatment[w0]/self.n_obs[w0])
        
        tau_obj = {"ate":tau,
                   "se":tau_se}
        
        return tau_obj

    def _calculate_average_treatment_effect_via_regression(self):
        """
        Regress Y on W and a constant and use heteroskedasticity-robust standard error.
        The ATE is the coef on W.
        """        
        tau = self.regression_results['coef'].loc['W']
        tau_se = self.regression_results['se'].loc['W']
        
        tau_obj = {"ate":tau,
                   "se":tau_se}        
        
        return tau_obj

    # --------------------
    # Public functions
    # --------------------
    def fit(self,Y,W):

        # Preprocess data
        super().preprocess_data(Y=Y,W=W)
        
        if self.use_regression:
            
            self.ols = OLS(include_constant=True,
                      covariance_type="HC1")
            
            self.regression_results = self.ols.fit(Y=Y,X=W)
            
        else:
            # Find mean and standard error for each treatment arm
            self.mean_outcome_per_treatment = Y.groupby(by=W,as_index=True).mean()
            self.var_outcome_per_treatment = Y.groupby(by=W,as_index=True).var()
            
        return self
            
    def calculate_heterogeneous_treatment_effect(self):
        raise CateError
    
    def calculate_average_treatment_effect(self, **kwargs):
        """
        """
        if self.use_regression:
            tau_obj = self._calculate_average_treatment_effect_via_regression()
        else:
            tau_obj = self._calculate_average_treatment_effect_via_differences(**kwargs)
            
        return tau_obj

        

