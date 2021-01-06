# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
import math

# CustomController class (inherits from BaseController)
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

        # Add additional member variables according to your need here.
        
        self.diff_curr=0
        self.diff_accum=0
        self.diff_prev=0
        
        self.X_prev=0
        self.Y_prev=0
        self.X_curr=0
        self.Y_curr=0
        self.v_error_curr=0
        self.v_error_prev=0
        self.v_error_accum=0
         

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        kP_lat=1
        kI_lat=1.5
        kD_lat=0
        
        CTE_curr=0
        Node_closest=0
        
        LH=48
        
        
        CTE_curr,Node_closest=closestNode(X,Y,trajectory)
        
  
        
        #Determining sign of CTE
        X1_path=trajectory[Node_closest][0]
        Y1_path=trajectory[Node_closest][1]  
        X2_path=trajectory[Node_closest+LH][0]
        Y2_path=trajectory[Node_closest+LH][1]
        traj_angle=np.arctan2((Y2_path-Y1_path),(X2_path-X1_path))
        angle_diff=wrapToPi(traj_angle-psi)
       
    
    
        
        self.diff_prev=self.diff_curr
        self.diff_curr=angle_diff
        
        
        
        diff_rate=(self.diff_curr-self.diff_prev)/delT
        
        

        delta=(kP_lat*self.diff_curr)+(kI_lat*self.diff_accum)+(kD_lat*diff_rate)
        
        if(np.absolute(delta)>np.pi/6):
            if delta<0:
                delta=-np.pi/6
            else:
                delta=np.pi/6
        if(np.absolute(delta)>np.pi/6):
            print('exceeded')

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        
        v_des=5.3 #m/s
        
        kP_lon=.5
        kI_lon=0
        kD_lon=0
                       
        self.X_prev=X
        self.Y_prev=Y

        self.X_prev=self.X_curr
        self.Y_prev=self.Y_curr
        self.X_curr=X
        self.Y_curr=Y
        
        distance=np.sqrt(((self.X_curr-self.X_prev)**2)+((self.Y_curr-self.Y_prev)**2))
        
        v=distance/delT
        
        v_error_curr=v_des-v
     
        self.v_error_prev=self.v_error_curr
        self.v_error_curr=v_error_curr
        self.v_error_accum=self.v_error_accum+v_error_curr
        
        self.v_error_accum=self.v_error_accum+v_error_curr
        v_error_rate=(self.v_error_curr-self.v_error_prev)/delT
        
        
        F=self.m/delT*(kP_lon*self.v_error_curr+kI_lon*self.v_error_accum+kD_lon*v_error_rate)
        #print(v)
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
