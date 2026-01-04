import numpy as np 

from scipy import stats


class KL_compute:
    def __init__(self):
        pass
    def compute( self,y_1, y_2):
        
        
        return np.mean(np.log(y_1 )-  np.log( y_2 )) 
        
    
    
    
class kernel_density():
    def __init__(self,data):
        self.kernel = stats.gaussian_kde(np.array(data).transpose())
    def compute(self, X_new):
        return self.kernel(np.array(X_new).transpose())
