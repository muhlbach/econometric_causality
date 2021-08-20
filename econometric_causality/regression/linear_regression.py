#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd
import statsmodels.api as sm
from warnings import warn

# User
from utils.tools import break_links
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class OLS(object):
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 include_constant=True,
                 covariance_type="HC1",
                 ):
        # Initialize inputs
        self.include_constant = include_constant
        self.covariance_type = covariance_type
        
        # Sanity check parameters
        if self.covariance_type not in self.COVARIANCE_TYPE_AVAILABLE:
            raise Exception(f"""
                            Argument 'covariance_type' is currently {self.covariance_type}.
                            However, only the following options are available: {self.COVARIANCE_TYPE_AVAILABLE}
                            """)
                            
    # --------------------
    # Class variables
    # --------------------
    COVARIANCE_TYPE_AVAILABLE = ['nonrobust','HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'cluster']

    # --------------------
    # Private functions
    # --------------------

    # --------------------
    # Public functions
    # --------------------
    def fit(self,Y,X):
        X = break_links(x=X)

        if self.include_constant:
            X = sm.add_constant(X, prepend=True)
        else:
            warn("No constant will be added. Make sure data 'X' contain a constant")
            
        # Set up model
        model = sm.OLS(endog=Y,
                       exog=X,
                       missing="none",
                       hasconstant=True
                       )
        
        # Fit model
        try:
            results = model.fit(method='qr',
                                cov_type=self.covariance_type,
                                cov_kwds=None,
                                use_t=True)
        except:
            results = model.fit(method='pinv',
                                cov_type=self.covariance_type,
                                cov_kwds=None,
                                use_t=True)
        
        # Extract results
        beta_hat = results.params
        se_hat = np.sqrt(np.diag(results.cov_params()))
        
        if isinstance(beta_hat, pd.Series):
            se_hat = pd.Series(se_hat, index=beta_hat.index)
        
        t_stat = results.tvalues
        p_value = results.pvalues
        
        # Construct dictionary to be returned
        obj_returned = {
            "coef":beta_hat,
            "se":se_hat,
            "t_stats":t_stat,
            "p_values":p_value,
            }
        
        return obj_returned


