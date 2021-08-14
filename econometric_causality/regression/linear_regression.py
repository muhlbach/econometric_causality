#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class OLS(object):
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 method="matrix_inversion",
                 include_constant=True,
                 covariance_type="standard",
                 ):
        # Initialize inputs
        self.method = method
        self.include_constant = include_constant
        self.covariance_type = covariance_type

    # --------------------
    # Class variables
    # --------------------

    # --------------------
    # Private functions
    # --------------------

    # --------------------
    # Public functions
    # --------------------
    def fit(self,Y,X):
        
        if self.include_constant:
            X["cons"] = 1
            
        if self.method=="matrix_inversion":
            beta, ssr, rank, s = np.linalg.lstsq(a=X, b=Y,rcond=None)
            residuals = Y.sub(X.dot(beta).squeeze(), axis=0)
            
            

        if self.covariance_type=="standard":            
            sigma2_hat = np.dot(residuals.T,residuals) / (X.shape[0]-X.shape[1])
            vcov_beta_hat = sigma2_hat * np.linalg.inv(np.dot(X.T, X))
            
        # Estimate of standard errors
        se_beta_hat = np.sqrt(np.diag(vcov_beta_hat))

        beta = pd.Series(beta.ravel(), index=X.columns)
        se_beta_hat = pd.Series(se_beta_hat.ravel(), index=X.columns)
        
        return beta,se_beta_hat
