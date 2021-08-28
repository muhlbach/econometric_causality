#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# User
from base.base_estimator import BaseCateEstimator
from base.propensity_score import BasePropensityScoreEstimator
from utils.exceptions import CateError, DfError
from utils.sanity_check import check_propensity_score_estimator
#------------------------------------------------------------------------------
# Treatment Effect Estimators
#------------------------------------------------------------------------------
class TreatmentEffectEstimator(BaseCateEstimator,BasePropensityScoreEstimator):
    """
    This class estimates treatment effects based on matching on inputs
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 n_boostrap_samples=1000,
                 verbose=False
                 ):
        # Initialize inputs
        self.n_boostrap_samples = n_boostrap_samples
        self.verbose = verbose

    # --------------------
    # Class variables
    # --------------------


    # --------------------
    # Private functions
    # --------------------
    def _compute_ipw_means(self,df):
        
        if not isinstance(df, pd.DataFrame):
            raise DfError(df)
        
        # Find unique treatments
        unique_treatments = np.sort(df["W"].unique()).tolist()
        
        # Initialize            
        mean_weighted_outcome_per_treatment = pd.DataFrame(index=[0],columns=unique_treatments)
            
        for w in unique_treatments:
            mean_weighted_outcome_per_treatment.loc[:,w] = (((df["W"]==w)*df["Y"])/df["propensity_score_realized"]).mean()
                            
        return mean_weighted_outcome_per_treatment


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
            propensity_score_raw = propensity_score_estimator.predict_proba(X=X)
        except AttributeError as attribute_error_message:
            if self.verbose:                
                print(f"""
                      AttributeError caught when calling 'predict_proba' on {type(propensity_score_estimator).__name__}.
                      Defaulting to 'predict'.
                      The original error message was: 
                      {str(attribute_error_message)}
                      """)
            propensity_score_raw = propensity_score_estimator.predict(X=X)
            
        # Transform to df
        propensity_score_raw = pd.DataFrame(propensity_score_raw)
                                        
        # Make sure probabilities are well-defined, e.g. they must sum to 1 over the treatment arms
        if propensity_score_raw.shape[1]>1:
           propensity_score_raw = propensity_score_raw.div(propensity_score_raw.sum(axis=1), axis=0)
             
        # TODO: Make this a robustness check instead
        # Correction
        propensity_score_raw.mask(cond=(propensity_score_raw < self.LOWER_LIMIT_PROPENSITY_SCORE) | (propensity_score_raw > self.UPPER_LIMIT_PROPENSITY_SCORE),
                              other=np.nan,
                              inplace=True)    
               
        # Probability of receiving the last treatment
        self.propensity_score = propensity_score_raw.iloc[:,-1]
        
        # Pre-allocate realized treatment, i.e. P(W=w|X=x)
        self.propensity_score_realized = pd.Series(index=W.index, name="propensity_score_realized", dtype="float64")

        # Convert propensity score to the "realized" treatment, meaning it should be a vector that represents the specific treatment
        if propensity_score_raw.shape[1]>1:
            propensity_score_raw.columns = self.unique_treatments
            
            for w in self.unique_treatments:
                mask = (W==w)
                self.propensity_score_realized.loc[mask] = propensity_score_raw.loc[mask,w]
        else:
            # Implicitly, we assume the last unique treatment is "highest" value, e.g., 0 vs 1
            self.propensity_score_realized[:] = np.where(W==self.unique_treatments[-1],
                                                         propensity_score_raw.squeeze(),
                                                         1-propensity_score_raw.squeeze())
        
        # Estimate the ATE between first and last treatment arm
        self.weighted_difference_bt_first_last_treatment = (Y*(W-self.propensity_score)) / (self.propensity_score*(1-self.propensity_score))

        # Collect data
        df = pd.concat([Y,W,self.propensity_score_realized], axis=1)

        # Compute weighted means
        self.mean_weighted_outcome_per_treatment = self._compute_ipw_means(df=df).T.squeeze().to_dict()

        _, self.var_weighted_outcome_per_treatment = super().bootstrap_mean_se(df=df,
                                                                               mean_estimator=self._compute_ipw_means,
                                                                               n_boostrap_samples=100)
        
        """
        Note: We cannot take the groupwise average. We need W to be multiplied on the Y. 
        That is, the code below will not provide a consistent estimate
        
        # self.mean_weighted_outcome_per_treatment = Y.div(self.propensity_score_realized).groupby(by=W,
        #                                                                                          as_index=True).mean().to_dict()
        """
        
        return self

    def calculate_heterogeneous_treatment_effect(self):
        raise CateError
    
    def calculate_average_treatment_effect(self,w0=None,w1=None):
        
        if all([w is None for w in [w0,w1]]):
            w0 = self.unique_treatments[0]
            w1 = self.unique_treatments[1]
        
        # Compute the ATE as the difference in sample means of the outcome between group1 (treated) and group2 (control)
        tau = self.mean_weighted_outcome_per_treatment[w1] - self.mean_weighted_outcome_per_treatment[w0]
        tau_se = np.sqrt(self.var_weighted_outcome_per_treatment[w1]/self.n_obs[w1] + self.var_weighted_outcome_per_treatment[w0]/self.n_obs[w0])
        
        tau_obj = {"ate":tau,
                   "se":tau_se}
    
        return tau_obj
