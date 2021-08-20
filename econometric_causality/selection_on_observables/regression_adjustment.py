#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd

# User
from base.base_estimator import BaseCateEstimator
from regression.linear_regression import OLS
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
    def fit(self,Y,W,X):

        # Preprocess data
        super().preprocess_data(Y=Y,W=W)

        # Instantiate regression
        ols = OLS(include_constant=True,
                  covariance_type="HC1")
        
        # Combine data
        WX = pd.concat([W,X], axis=1)
        
        # Fit regression
        self.regression_results = ols.fit(Y=Y,X=WX)
            
        return self

    def calculate_heterogeneous_treatment_effect(self):
        raise CateError
    
    def calculate_average_treatment_effect(self):
        """
        """
        tau_obj = {
            "ate":self.regression_results["coef"].loc[self.W_name],
            "se":self.regression_results["se"].loc[self.W_name],
            "t_stat":self.regression_results["t_stats"].loc[self.W_name],
            "p_value":self.regression_results["p_values"].loc[self.W_name],
            }
            
        return tau_obj
    
    