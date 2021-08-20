#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

# User
from base.base_estimator import BaseCateEstimator
from utils.exceptions import CateError
from utils.sanity_check import check_propensity_score_estimator
#------------------------------------------------------------------------------
# Treatment Effect Estimators
#------------------------------------------------------------------------------
class MatchingOnInput(BaseCateEstimator):
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
    def fit(self,Y,W,I):
        """
        Parameters
        ----------
        Y : pd.Series
            Outcome variable
        W : pd.Series
            Treatment variable
        I : pd.Dataframe
            Input variable to match on
        """
        # Preprocess data
        super().preprocess_data(Y=Y,W=W)
        
        # Check input type
        if isinstance(I, pd.Series):
            I = I.to_frame()

        # Instantiate object to identofy the nearest neighbors
        nearestneighbors = NearestNeighbors(n_neighbors=1,
                                            radius=1.0,
                                            algorithm='auto',
                                            leaf_size=30,
                                            metric='minkowski',
                                            p=2,
                                            metric_params=None,
                                            n_jobs=None)
        
        # Initialize matched outcomes
        Y_matched = pd.Series(index=Y.index, dtype=Y.dtype, name=Y.name)
        
        # For each treatment arm, find the nearest neighbor in the other treatment arm        
        for w in self.unique_treatments:
                        
            # Mask treatment w
            mask = self.mask_treatment[w]
            
            # Fit on treatment w
            nearestneighbors.fit(X=I[mask])
                                    
            # Find neighbors among treatment ~w
            neigbors_idx = nearestneighbors.kneighbors(X=I[~mask], return_distance=False).flatten()
            
            # Use outcomes values that match based on X
            Y_matched[~mask] = Y.loc[neigbors_idx].values

        self.mean_residualized_outcome_per_treatment = {}        
        for w in self.unique_treatments:
            self.mean_residualized_outcome_per_treatment[w] = ((W==w) * (Y - Y_matched)).mean()
            
        return self

    def calculate_heterogeneous_treatment_effect(self):
        raise CateError
    
    def calculate_average_treatment_effect(self,w0=None,w1=None):
        
        if all([w is None for w in [w0,w1]]):
            w0 = self.unique_treatments[0]
            w1 = self.unique_treatments[1]
        
        # Compute the ATE as the difference in sample means of the outcome between group1 (treated) and group2 (control)
        tau = self.mean_residualized_outcome_per_treatment[w1] - self.mean_residualized_outcome_per_treatment[w0]
        
        tau_obj = {"ate":tau}
        
        return tau_obj



#------------------------------------------------------------------------------
# Treatment Effect Estimators
#------------------------------------------------------------------------------
class MatchingOnCovariates(MatchingOnInput):
    """
    This class estimates treatment effects based on matching on covariates
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 verbose=False
                 ):
        # Initialize inputs
        self.verbose = verbose
        super().__init__(verbose=self.verbose)

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

        super().fit(Y=Y, W=W, I=X)
            
        return self

    def calculate_heterogeneous_treatment_effect(self):
        super().calculate_heterogeneous_treatment_effect()
    
    def calculate_average_treatment_effect(self,w0=None,w1=None):
        return super().calculate_average_treatment_effect()
    

class MatchingOnPropensityScore(MatchingOnInput):
    """
    This class estimates treatment effects based on matching on covariates
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 verbose=False
                 ):
        # Initialize inputs
        self.verbose = verbose
        super().__init__(verbose=self.verbose)

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
        
        super().fit(Y=Y, W=W, I=propensity_score)
            
        return self

    def calculate_heterogeneous_treatment_effect(self):
        super().calculate_heterogeneous_treatment_effect()
    
    def calculate_average_treatment_effect(self,w0=None,w1=None):
        return super().calculate_average_treatment_effect()
  
    
    