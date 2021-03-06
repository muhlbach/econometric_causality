#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
#------------------------------------------------------------------------------
# Empty class
#------------------------------------------------------------------------------
class BaseCateEstimator(ABC):
    """Base class for all CATE estimators in this package."""
    # --------------------
    # Constructor function
    # --------------------
    # No __init__

    # --------------------
    # Class variables
    # --------------------
    # No class variables

    # --------------------
    # Private functions
    # --------------------
    # No private functions

    # --------------------
    # Public functions
    # --------------------
    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Estimate the counterfactual model from data, i.e. estimates functions
        :math:`\\tau(X, W0, W1)`, :math:`\\partial \\tau(W, X)`.

        Note that the signature of this method may vary in subclasses (e.g. classes that don't
        support instruments will not allow a `Z` argument)

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        W: (n, d_w) matrix or vector of length n
            Treatments for each sample
        X: optional (n, d_x) matrix
            Features for each sample (include both candidates for heterogeneity and standard controls)
        Z: optional (n, d_z) matrix
            Instruments for each sample
        
        Returns
        -------
        self

        """
        pass

    @abstractmethod
    def calculate_heterogeneous_treatment_effect(self, X=None, *, W0, W1):
        """
        Calculate the heterogeneous treatment effect :math:`\\tau(X, W0, W1)`.

        The effect is calculated between the two treatment points
        conditional on a vector of features on a set of m test samples :math:`\\{W0_i, W1_i, X_i\\}`.

        Parameters
        ----------
        W0: (m, d_w) matrix or vector of length m
            Base treatments for each sample
        W1: (m, d_w) matrix or vector of length m
            Target treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        ??: (m, d_y) matrix
            Heterogeneous treatment effects on each outcome for each sample
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        pass
    
    
    def calculate_average_treatment_effect(self, X=None, *, W0, W1):
        """
        Calculate the average treatment effect :math:`E_X[\\tau(X, W0, W1)]`.

        The effect is calculated between the two treatment points and is averaged over
        the population of X variables.

        Parameters
        ----------
        W0: (m, d_w) matrix or vector of length m
            Base treatments for each sample
        W1: (m, d_w) matrix or vector of length m
            Target treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        ??: float or (d_y,) array
            Average treatment effects on each outcome
            Note that when Y is a vector rather than a 2-dimensional array, the result will be a scalar
        """
        return np.mean(self.calculate_heterogeneous_treatment_effect(X=X, W0=W0, W1=W1), axis=0)   


    def preprocess_data(self,Y,W, *args):
        
        # TODO: Sanity check data
        
        # Save data
        self.Y = Y
        self.W = W
                
        self.W_name = W.name
        
        # List unique treatment
        self.unique_treatments = np.sort(W.unique()).tolist()
        
        # Find masks
        self.mask_treatment = {}
        self.n_obs = {}
        for w in self.unique_treatments:
            self.mask_treatment[w] = (W==w)
            self.n_obs[w] = self.mask_treatment[w].sum()     
            
        self.N = len(Y)
            

    def bootstrap_mean_se(self, df, mean_estimator, n_boostrap_samples=100):
        """
        Bootstrap mean and variance given as function that produces an estimate from a sample        
        """
        # Initialize 
        df_bootstrap = pd.DataFrame()
        
        # Number of samples
        n_obs = df.shape[0]
            
        for _ in range(n_boostrap_samples):
            
            # Draw bootstrap sample with replacement
            df_b = df.sample(n=n_obs, replace=True)
            
            df_bootstrap = df_bootstrap.append(mean_estimator(df_b), ignore_index=True)
          
       # Compute mean and variance across bootstrap samples 
        bootstrapped_mean, bootstrapped_variance = df_bootstrap.mean(axis=0).to_dict(), df_bootstrap.var(axis=0).to_dict()

        return bootstrapped_mean, bootstrapped_variance





    