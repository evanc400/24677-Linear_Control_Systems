# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

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
        
        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep, driver):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)

        # You are free to reuse or refine your code from P3 in the spaces below.

        # ---------------|Lateral Controller|-------------------------
         # ---------------|Lateral Controller|-------------------------
        # Use the results of linearization to create a state-space model
        A = np.array([[0,1,0,0],[0,-4*Ca/(m*xdot),4*Ca/m,2*Ca*(lr-lf)/(m*xdot)],
            [0,0,0,1], [0,(2*Ca)*(lr-lf)/(Iz*xdot), (2*Ca)*(lf-lr)/Iz, \
            (-2*Ca)*(lf**2 + lr**2)/(Iz*xdot)]])
        B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])
        C = np.eye(4)
        D = np.zeros((4,1))
        
        system = signal.StateSpace(A, B, C, D)
        discrete = system.to_discrete(delT)
        
        Ad=discrete.A
        Bd=discrete.B
        
        
        #LQR
        
        q_e1=10
        q_e1_dot=2
        q_e2=1
        q_e2_dot=14
        
        Q=np.array([[q_e1,0,0,0],
                    [0,q_e1_dot,0,0],
                    [0,0,q_e2,0],
                    [0,0,0,q_e2_dot]])
                    
        
        R=np.array([13.5])
        
        S = np.matrix(linalg.solve_discrete_are(Ad, Bd, Q, R))
        K = np.matrix(linalg.inv(B.T@S@B+R)@(B.T@S@A))
        ##
        
        # Find the closest node to the vehicle
        _, node = closestNode(X, Y, trajectory)

        # Choose a node that is ahead of our current node based on index
        forwardIndex = 214

        # Determine desired heading angle and e1 using two nodes - one ahead, and one closest
        # We use a try-except so we don't attempt to grab an index that is out of scope

        # To define our error-based states, we use definitions from documentation.
        # Please see page 34 of Rajamani Rajesh's book "Vehicle Dynamics and Control", which
        # is available online through the CMU library, for more information.
        # It is important to note that numerical derivatives of e1 and e2 will also work well.
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-trajectory[node,1], trajectory[node+forwardIndex,0]-trajectory[node,0])
            e1 =  (Y - trajectory[node+forwardIndex,1])*np.cos(psiDesired) - (X - trajectory[node+forwardIndex,0])*np.sin(psiDesired)

        except:
            psiDesired = np.arctan2(trajectory[-1,1]-trajectory[node,1], trajectory[-1,0]-trajectory[node,0])
            e1 =  (Y - trajectory[-1,1])*np.cos(psiDesired) - (X - trajectory[-1,0])*np.sin(psiDesired)

        e1dot = ydot + xdot*wrapToPi(psi - psiDesired)
        e2 = wrapToPi(psi - psiDesired)
        e2dot = psidot # This definition would be psidot - psidotDesired if calculated from curvature

        # Assemble error-based states into array
        states = np.array([e1,e1dot,e2,e2dot])

        # Calculate delta via u = -Kx
        delta = float(-K @ states)   



        # ---------------|Longitudinal Controller|-------------------------
        # PID gains
        kp = 200
        ki = 10
        kd = 30

        # Reference value for PID to tune to
        desiredVelocity = 14

        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError

        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT


        
        # Setting brake intensity is enabled by passing
        # the driver object, which is used to provide inputs
        # to the car, to our update function
        # Using this function is purely optional.
        # An input of 0 is no brakes applied, while
        # an input of 1 is max brakes applied

        # driver.setBrakeIntensity(clamp(someValue, 0, 1))

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta