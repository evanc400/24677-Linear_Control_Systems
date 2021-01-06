#P2

# Fill the respective function to implement the PID controller
# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# Custom Controller Class
class CustomController(BaseController):
    def __init__(self, trajectory):
        
        super().__init__(trajectory)
        
        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
   
        self.previousPsiError = 0
        self.previousCTE=0
        self.previousXdotError = 0
        
        
        
        
        

    def update(self, timestep):
        
        trajectory = self.trajectory
        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
        A=np.array([[0,1,0,0],
                [0,-(4*self.Ca)/(self.m*xdot),(4*self.Ca)/self.m,-(2*self.Ca*(self.lf-self.lr))/(self.m*xdot)],
                [0,0,0,1],
                [0,-(2*self.Ca*(self.lf-self.lr))/(self.Iz*xdot),(2*self.Ca*(self.lf-self.lr))/self.Iz,-1*(2*self.Ca*(self.lf**2+self.lr**2))/(self.Iz*xdot)]])
        
        B=np.array([[0],
                    [(2*self.Ca)/self.m],
                    [0],
                    [(2*self.Ca*self.lf)/self.Iz]])
                    
          
        # ---------------|Lateral Controller|-------------------------
        # Find the closest node to the vehicle
        CTE, node = closestNode(X, Y, trajectory) #e1: CTE
        
        # Choose a node that is ahead of our current node based on index
        forwardIndex = 50
        # Two distinct ways to calculate the desired heading angle:
        # 1. Find the angle between a node ahead and the car's current,→ position
        # 2. Find the angle between two nodes - one ahead, and one closest
        # The first method has better overall performance, as the second,→ method
        
        # can read zero error when the car is not actually on the,→ trajectory
        # We use a try-except so we don't attempt to grab an index that is,→ out of scope
        # 1st method
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-Y, \
            trajectory[node+forwardIndex,0]-X)
        except:
            psiDesired = np.arctan2(trajectory[-1,1]-Y, \
            trajectory[-1,0]-X)
            
        
        
        

        
        # Error states
        psiError = wrapToPi(psi-psiDesired) #e2
        psiError_dot = wrapToPi((psiError - self.previousPsiError)/delT) #e2_dot          
        
        
        if psiError<0:
            CTE=-1*CTE
        
        
        CTE_dot = (CTE-self.previousCTE)/delT #e1_dot
        #print("Previous CTE: ", self.previousCTE)
        #print("Current CTE: ", CTE)
        #print("Previous psi error: ", self.previousPsiError )
        #print("Current psi: ", psiError)
        
        errorVector=np.array([[CTE],[CTE_dot],[psiError],[psiError_dot]])
        
        P= np.array([-10, -2, -0.1-1j, -0.1+1j])
        #Best so far: 18.6 (avg dist), 23.6 (max dist):  P= np.array([-60, -1, -0.001+1j, -0.001-1j])
        
        sys_placed_poles = signal.place_poles(A, B, P)
        K=sys_placed_poles.gain_matrix

        u = -K@errorVector
        delta = u[0][0]
        delta = wrapToPi(delta)
        
        #update previous:
        self.previousCTE=CTE
        self.previousPsiError=psiError
        

        # ---------------|Longitudinal Controller|-------------------------

        # PID gains
        kp = 200
        ki = 10
        kd = 30
        # Reference value for PID to tune to
       
        desiredVelocity = 6
        
        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError
        
        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta