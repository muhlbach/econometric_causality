#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class CateError(Exception):
    """
    Exception raised when heterogeneous treatment effects are not available
    """
    def __init__(self, message="Heterogeneous treatment effects are not available."):
        self.message = message
        super().__init__(self.message)    
