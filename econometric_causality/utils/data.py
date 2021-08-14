#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def get_colnames(x,prefix="X"):
    try:
        dim = x.shape[1]
        colnames = [prefix+str(j) for j in np.arange(start=1,stop=dim+1)]
    except IndexError:
        colnames = [prefix]
        
    return colnames
    
def convert_to_dfs(Y=None,W=None,X=None,Z=None):
    
    if Y is not None:
        if isinstance(Y, np.ndarray):
            Y = pd.DataFrame(Y, columns=get_colnames(x=Y,prefix="Y"))
            
    if W is not None:
        if isinstance(W, np.ndarray):
            W = pd.DataFrame(W, columns=get_colnames(x=W,prefix="W"))
            
    if X is not None:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=get_colnames(x=X,prefix="X"))
            
    if Z is not None:
        if isinstance(Z, np.ndarray):
            Z = pd.DataFrame(Z, columns=get_colnames(x=Z,prefix="Z"))
            
    return Y, W, X, Z
            

def generate_data_rct(N=1000, p=5, return_as_df=False, **kwargs):
    
    # Beta
    beta = kwargs.get('beta', np.ones(p))

    # Mean of X
    mu = kwargs.get('mu', np.zeros(p))
    
    # Covariance of X
    Sigma = kwargs.get('Sigma', np.identity(p))
    
    # Covariance of eps
    Gamma = kwargs.get('Gamma', np.identity(2))
    
    # Error term
    epsilon = np.random.multivariate_normal(mean=np.zeros(2), cov=Gamma, size=N)

    # Tau
    tau = kwargs.get('tau', 1)
    
    # Covariates
    X = np.random.multivariate_normal(mean=mu, cov=Sigma, size=N)

    # Treatment dummy
    W = np.random.binomial(n=1, p=0.5, size=N)
    
    # Potential outcomes
    Y_baseline = X @ beta
    Y0 = Y_baseline + epsilon[:,0]
    Y1 = tau + Y_baseline + epsilon[:,1]
    
    # Observed outcome
    Y = (1-W)*Y0 + W*Y1

    # Convert to dataframes
    Y, W, X, _ = convert_to_dfs(Y=Y,W=W,X=X)

    if return_as_df:
        df = pd.concat([Y,W,X], axis=1)
        
        return df
    else:
        return Y, W, X

        

        
                                  