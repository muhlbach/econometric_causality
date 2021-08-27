#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# User
from base.base_estimator import BaseCateEstimator
from utils.exceptions import CateError
# from utils.sanity_check import conditional_mean_estimator
#------------------------------------------------------------------------------
# Treatment Effect Estimators
#------------------------------------------------------------------------------
class TreatmentEffectEstimator(BaseCateEstimator):
    """
    This class estimates treatment effects based on matching on inputs
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 verbose=False,
                 homogeneous_slope_X=False
                 ):
        # Initialize inputs
        self.homogeneous_slope_X = homogeneous_slope_X
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
    def fit(self,Y,W,X,conditional_mean_estimator=RandomForestRegressor(n_estimators=100,
                                                                        criterion='mse',
                                                                        max_depth=None,
                                                                        min_samples_split=2,
                                                                        min_samples_leaf=1,
                                                                        min_weight_fraction_leaf=0.0,
                                                                        max_features='auto',
                                                                        max_leaf_nodes=None,
                                                                        min_impurity_decrease=0.0,
                                                                        min_impurity_split=None,
                                                                        bootstrap=True,
                                                                        oob_score=False,
                                                                        n_jobs=None,
                                                                        random_state=None,
                                                                        verbose=0,
                                                                        warm_start=False,
                                                                        ccp_alpha=0.0,
                                                                        max_samples=None)):
                
        # Preprocess data
        super().preprocess_data(Y=Y,W=W)
        
        # Estimate propensity scores
        # check_conditional_mean_estimator(estimator=conditional_mean_estimator)
        # conditional_mean_estimator.fit(X=X,y=W)    
        
        # HERE
            
        return self

    def calculate_heterogeneous_treatment_effect(self):
        raise CateError
    
    def calculate_average_treatment_effect(self,w0=None,w1=None):
                
        tau_obj = {"ate":1}
        
        return tau_obj
