import numpy as np

class EKF_SLAM():
    def __init__(self, init_mu, init_P, dt, W, V, n):
        """Initialize EKF SLAM

        Create and initialize an EKF SLAM to estimate the robot's pose and
        the location of map features

        Args:
            init_mu: A numpy array of size (3+2*n, ). Initial guess of the mean
            of state.
            init_P: A numpy array of size (3+2*n, 3+2*n). Initial guess of
            the covariance of state.
            dt: A double. The time step.
            W: A numpy array of size (3+2*n, 3+2*n). Process noise
            V: A numpy array of size (2*n, 2*n). Observation noise
            n: A int. Number of map features


        Returns:
            An EKF SLAM object.
        """
        self.mu = init_mu  # initial guess of state mean
        self.P = init_P  # initial guess of state covariance
        self.dt = dt  # time step
        self.W = W  # process noise
        self.V = V  # observation noise
        self.n = n  # number of map features


    def _f(self, x, u):
        """Non-linear dynamic function.

        Compute the state at next time step according to the nonlinear dynamics f.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            x_next: A numpy array of size (3+2*n, ). The state at next time step
        """
        X=x[0]
        Y=x[1]
        psi=x[2]


        dot_x=u[0]
        dot_y=u[1]
        dot_psi=u[2]


        f_X=X+self.dt*(dot_x*np.cos(psi)-dot_y*np.sin(psi))
        f_Y=Y+self.dt*(dot_x*np.sin(psi)+dot_y*np.cos(psi))
        f_psi=self._wrap_to_pi(psi+self.dt*dot_psi)  #angle

        #x_next=np.array([[f_X],[f_Y],[f_psi]])
        x_next=np.array([f_X,f_Y,f_psi])

        for i in range(3,3+2*self.n):
            x_next=np.append(x_next,np.array([x[i]]))

        return x_next


    def _h(self, x):
        """Non-linear measurement function.

        Compute the sensor measurement according to the nonlinear function h.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.

        Returns:
            y: A numpy array of size (2*n, ). The sensor measurement.
        """
        X=x[0]
        Y=x[1]
        psi=x[2]

        y= np.array([])

        #calculating the first half of y: the landmark distance measurements
        for i in range(3,3+2*self.n,2):
            h_dist=np.sqrt( ((x[i]-X)**2) + ((x[i+1]-Y)**2))
            y=np.append(y,np.array([h_dist]))

        #calculating the second half of y: the landmark bearing measurements
        for i in range(3,3+2*self.n,2):

            h_bear=self._wrap_to_pi(np.arctan2((x[i+1]-Y),(x[i]-X))-psi)
            y=np.append(y,h_bear)

        return y


    def _compute_F(self, u):
        """Compute Jacobian of f

        You will use self.mu in this function.

        Args:
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            F: A numpy array of size (3+2*n, 3+2*n). The jacobian of f evaluated at x_k.
        """
        dot_x=u[0]
        dot_y=u[1]
        dot_psi=u[2]

        #First row and initializing F
        F_newRow=np.array([1,0,self.dt*(-dot_x*np.sin(self.mu[2]) - dot_y*np.cos(self.mu[2]))])
        F_newRow=np.append(F_newRow,np.array([0 for i in range(3,2*self.n+3)]))
        F=np.array([F_newRow])

        #Second row, appending to F
        F_newRow=np.array([0,1,self.dt*(dot_x*np.cos(self.mu[2]) - dot_y*np.sin(self.mu[2]))])
        F_newRow=np.append(F_newRow,np.array([0 for i in range(3,2*self.n+3)]))
        F=np.append(F,np.array([F_newRow]),axis=0)

        #Third row
        F_newRow=np.array([0,0,1])
        F_newRow=np.append(F_newRow,np.array([0 for i in range(3,2*self.n+3)]))
        F=np.append(F,np.array([F_newRow]),axis=0)

        #Creating last 2*n rows
        F_zeros=np.zeros((2*self.n,3))
        F_indentity=np.identity(2*self.n)
        F_bottom=np.append(F_zeros,F_indentity, axis=1)

        #Appending bottom half
        F=np.append(F,F_bottom,axis=0)

        return F


    def _compute_H(self):
        """Compute Jacobian of h

        You will use self.mu in this function.

        Args:

        Returns:
            H: A numpy array of size (2*n, 3+2*n). The jacobian of h evaluated at x_k.
        """
        X=self.mu[0]
        Y=self.mu[1]

        # distance sensor
        H_dist=np.array([[]])

        for i in range(3,3+2*self.n,2):
            H_newRow=np.array([0 for i in range(0,2*self.n+3)]) #3+2*n cols


            divisor_dist=np.sqrt( ((self.mu[i]-X)**2) + ((self.mu[i+1]-Y)**2))
            #first 2 elements in every row are same formula
            H_newRow[0]=(X-self.mu[i])/divisor_dist
            H_newRow[1]=(Y-self.mu[i+1])/divisor_dist

            #other 2 non-zero args are follow a diagonal through H starting from 3 & 4th elements in first row
            H_newRow[i]=(self.mu[i]-X)/divisor_dist
            H_newRow[i+1]=(self.mu[i+1]-Y)/divisor_dist

            #Append row to H if H isn't empty, else initialize H as first row
            if(H_dist.size != 0):
                H_dist=np.append(H_dist,np.array([H_newRow]), axis=0)
            else:
                H_dist=np.append(H_dist,np.array([H_newRow]), axis=1)

        # bearing sensor
        H_bear=np.array([[]])

        for i in range(3,3+2*self.n,2): #n rows
            H_newRow=np.array([0 for i in range(0,2*self.n+3)])


            divisor_bear=((self.mu[i]-X)**2) + ((self.mu[i+1]-Y)**2)

            #first 3 elements in every row are same formula
            H_newRow[0]=(self.mu[i+1]-Y)/divisor_bear
            H_newRow[1]=(X-self.mu[i])/divisor_bear
            H_newRow[2]=-1

            #other 2 non-zero args are follow a diagonal through H starting from 3 & 4th elements in first row
            H_newRow[i]=(Y-self.mu[i+1])/divisor_bear
            H_newRow[i+1]=(self.mu[i]-X)/divisor_bear

            #Append row to H if H isn't empty, else initialize H as first row
            if(H_bear.size != 0):
                H_bear=np.append(H_bear,np.array([H_newRow]), axis=0)
            else:
                H_bear=np.append(H_bear,np.array([H_newRow]), axis=1)

        H=np.append(H_dist,H_bear,axis=0)

        return H


    def predict_and_correct(self, y, u):
        """Predice and correct step of EKF

        You will use self.mu in this function. You must update self.mu in this function.

        Args:
            y: A numpy array of size (2*n, ). The measurements according to the project description.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            self.mu: A numpy array of size (3+2*n, ). The corrected state estimation
            self.P: A numpy array of size (3+2*n, 3+2*n). The corrected state covariance
        """

        # compute F and H matrix

        #last_mu = self.mu



        #***************** Predict step *****************#
        # predict the state
        F=self._compute_F(u)
        self.mu=self._f(self.mu,u)
        H=self._compute_H()

        # predict the error covariance
        self.P=F@self.P@F.T+self.W

        #***************** Correct step *****************#
        # compute the Kalman gain
        L=self.P@H.T@np.linalg.inv(H@self.P@H.T+self.V)

        # update estimation with new measurement
        error=y-self._h(self.mu)
        for i in range(self.n,2*self.n): #wrapping only second half
            error[i]=self._wrap_to_pi(error[i])
        self.mu=self.mu+L@(error)
        self.mu[2]=self._wrap_to_pi(self.mu[2])

        # update the error covariance
        self.P=(np.identity(3+2*self.n)-L@H)@self.P


        return self.mu, self.P


    def _wrap_to_pi(self, angle):
        angle = angle - 2*np.pi*np.floor((angle+np.pi )/(2*np.pi))
        return angle


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = np.array([[0.,  0.],
                  [0.,  20.],
                  [20., 0.],
                  [20., 20.],
                  [0,  -20],
                  [-20, 0],
                  [-20, -20],
                  [-50, -50]]).reshape(-1)

    dt = 0.01
    T = np.arange(0, 20, dt)
    n = int(len(m)/2)
    W = np.zeros((3+2*n, 3+2*n))
    W[0:3, 0:3] = dt**2 * 1 * np.eye(3)
    V = 0.1*np.eye(2*n)
    V[n:,n:] = 0.01*np.eye(n)

    # EKF estimation
    mu_ekf = np.zeros((3+2*n, len(T)))
    mu_ekf[0:3,0] = np.array([2.2, 1.8, 0.])
    # mu_ekf[3:,0] = m + 0.1
    mu_ekf[3:,0] = m + np.random.multivariate_normal(np.zeros(2*n), 0.5*np.eye(2*n))
    init_P = 1*np.eye(3+2*n)

    # initialize EKF SLAM
    slam = EKF_SLAM(mu_ekf[:,0], init_P, dt, W, V, n)

    # real state
    mu = np.zeros((3+2*n, len(T)))
    mu[0:3,0] = np.array([2, 2, 0.])
    mu[3:,0] = m

    y_hist = np.zeros((2*n, len(T)))
    for i, t in enumerate(T):
        if i > 0:
            # real dynamics
            u = [-5, 2*np.sin(t*0.5), 1*np.sin(t*3)]
            # u = [0.5, 0.5*np.sin(t*0.5), 0]
            # u = [0.5, 0.5, 0]

            mu[:,i] = slam._f(mu[:,i-1], u) + \
                np.random.multivariate_normal(np.zeros(3+2*n), W)

            # measurements
            y = slam._h(mu[:,i]) + np.random.multivariate_normal(np.zeros(2*n), V)
            y_hist[:,i] = (y-slam._h(slam.mu))
            # apply EKF SLAM
            mu_est, _ = slam.predict_and_correct(y, u)
            mu_ekf[:,i] = mu_est


    plt.figure(1, figsize=(10,6))
    ax1 = plt.subplot(121, aspect='equal')
    ax1.plot(mu[0,:], mu[1,:], 'b')
    ax1.plot(mu_ekf[0,:], mu_ekf[1,:], 'r--')
    mf = m.reshape((-1,2))
    ax1.scatter(mf[:,0], mf[:,1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2 = plt.subplot(322)
    ax2.plot(T, mu[0,:], 'b')
    ax2.plot(T, mu_ekf[0,:], 'r--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('X')

    ax3 = plt.subplot(324)
    ax3.plot(T, mu[1,:], 'b')
    ax3.plot(T, mu_ekf[1,:], 'r--')
    ax3.set_xlabel('t')
    ax3.set_ylabel('Y')

    ax4 = plt.subplot(326)
    ax4.plot(T, mu[2,:], 'b')
    ax4.plot(T, mu_ekf[2,:], 'r--')
    ax4.set_xlabel('t')
    ax4.set_ylabel('psi')

    plt.figure(2)
    ax1 = plt.subplot(211)
    ax1.plot(T, y_hist[0:n, :].T)
    ax2 = plt.subplot(212)
    ax2.plot(T, y_hist[n:, :].T)

    plt.show()
