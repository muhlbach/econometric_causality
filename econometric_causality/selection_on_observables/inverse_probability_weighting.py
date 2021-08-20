#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# User
from base.base_estimator import BaseCateEstimator
from utils.exceptions import CateError
from utils.sanity_check import check_propensity_score_estimator
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
    def fit(self,Y,W,X,propensity_score_estimator=LogisticRegression(penalty='l2',
                                                                     dual=False,
                                                                     tol=0.0001,
                                                                     C=1.0,
                                                                     fit_intercept=True,
                                                                     intercept_scaling=1,
                                                                     class_weight=None,
                                                                     random_state=None,
                                                                     solver='lbfgs',
                                                                     max_iter=100,
                                                                     multi_class='auto',
                                                                     verbose=0,
                                                                     warm_start=False,
                                                                     n_jobs=None,
                                                                     l1_ratio=None)):    
                
        # Preprocess data
        super().preprocess_data(Y=Y,W=W)
        
        # Estimate propensity scores
        check_propensity_score_estimator(estimator=propensity_score_estimator)
        propensity_score_estimator.fit(X=X,y=W)    
        
        # Some sklearn modules with have 'predict_proba' as a method. Try this before defaulting to 'predict'
        try:
            propensity_score = propensity_score_estimator.predict_proba(X=X)[:,-1]
        except AttributeError as attribute_error_message:
            if self.verbose:                
                print(f"""
                      AttributeError caught when calling 'predict_proba' on {type(propensity_score_estimator).__name__}.
                      Defaulting to 'predict'.
                      The original error message was: 
                      {str(attribute_error_message)}
                      """)
            propensity_score = propensity_score_estimator.predict(X=X)
            
        # Transform to series
        propensity_score = pd.Series(propensity_score)

        # TODO: Make this a robustness check instead
        # Correction
        propensity_score = np.where(propensity_score>0.99,
                                    np.nan,
                                    np.where(propensity_score<0.01,
                                             np.nan,
                                             propensity_score)
                                    )

        # Compute the estimated probablity for receiving the treatment that the individual actually received
        propensity_score_realized = np.where(W==self.unique_treatments[-1], propensity_score, 1-propensity_score)
        
        self.mean_weighted_outcome_per_treatment = {}        
        for w in self.unique_treatments:
            self.mean_weighted_outcome_per_treatment[w] = ((W==w)*Y/propensity_score_realized).mean()
            
        # Note: The above corresponds to the simple difference between averages:
        #W=1: (W * Y / propensity_score).mean()
        #W=0: ((1-W) * Y / (1-propensity_score)).mean()
            
        return self

    def calculate_heterogeneous_treatment_effect(self):
        raise CateError
    
    def calculate_average_treatment_effect(self,w0=None,w1=None):
        
        if all([w is None for w in [w0,w1]]):
            w0 = self.unique_treatments[0]
            w1 = self.unique_treatments[1]
        
        # Compute the ATE as the difference in sample means of the outcome between group1 (treated) and group2 (control)
        tau = self.mean_weighted_outcome_per_treatment[w1] - self.mean_weighted_outcome_per_treatment[w0]
        
        tau_obj = {"ate":tau}
        
        return tau_obj
