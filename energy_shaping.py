import numpy as np
import scipy 

class EnergyShapingController():
    def __init__(self, mass, length, damping, torque_limit, gravity, K):
        self.K = K
        self.m = mass
        self.l = length
        self.g = gravity
        self.b = damping
        self.torque_limit = torque_limit

    def get_control_output(self, x):
        # Create a Energy Shaping Controller for swing up of simple pendulum 
        Ted = self.m * self.l * self.g  # desired energy
        Tec = (self.m * (self.l * x[1])**2)/2 - self.m * self.l * self.g * np.cos(x[0]) # current energy
        tau_es = -self.K * x[1] * (Tec - Ted) + self.b*x[1]
        tau = np.clip(tau_es,-self.torque_limit, self.torque_limit)
        
        return tau

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    #Solve the Algebraic Riccati Equation
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)

    #compute the LQR gain
    K = scipy.linalg.inv(R).dot(B.T.dot(S))
    eigVals, eigVecs = scipy.linalg.eig(A-B.dot(K))
    return K, S, eigVals
    
class LQRController():
    def __init__(self, mass, length, gravity, damping, torque_limit, Q, R):
        self.Q = Q                 # 2x2 Matrix cost on state error
        self.R = R                 # 1x1 Matrix cost on actuation
        self.m = mass
        self.l = length
        self.g = gravity
        self.b = damping
        self.torque_limit = torque_limit
        A, B = self.linearize()
        self.K = np.zeros((2, 2))
        # Continuous Time LQR
        self.K, self.S, _ = lqr(A, B, Q, R)
        print("LQR Gain Matrix (K):", self.K)
        # Discrete Time LQR
        # self.K, _, _ = dlqr(A, B, Q, R)
    
    def linearize(self):
        # Linearized equations of motion for simple pendulum
        # at the upright/topmost state (pi, 0)
        # Either Continuous form: x_dot = Ax + Bu
        # Or discrete form: x[n+1] = Ax[n] + Bu[n]

        ### Type here!
        
        A = np.array([[0, 1], [self.g/self.l , -self.b/(self.m*(self.l)**2)]])
        B = np.array([[0],[1 / (self.m * (self.l)**2)]])

        ###
        
        return A, B

    def get_control_output(self, x):
        # Use the LQR Gain matrix K to create the controller.
        tau = -self.K.dot(x - np.array([np.pi, 0]))
        # Add a torque limit before returning the tau value.
        return np.clip(tau[0], -self.torque_limit, self.torque_limit)
        
        
class EnergyShapingAndLQRController():
    """
    Controller which swings up the pendulum with the energy shaping
    controller and stabilizes the pendulum with the lqr controller.
    """
    def __init__(self, mass=1.0, length=0.5, damping=0.1,
                 gravity=9.81, torque_limit=np.inf, K=1.0,
                 Q=np.diag((10, 1)), R=np.array([[1]]), compute_RoA=False):
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity

        self.energy_shaping_controller = EnergyShapingController(mass=mass,
                                                                 length=length,
                                                                 damping=damping,
                                                                 gravity=gravity,
                                                                 torque_limit=torque_limit,
                                                                 K=K)
        self.lqr_controller = LQRController(mass=mass,
                                            length=length,
                                            damping=damping,
                                            gravity=gravity,
                                            torque_limit=torque_limit,
                                            Q=Q,
                                            R=R)

        self.active_controller = "none"
        self.swingup_time = None
        
        self.meas_time = 0.0

    def set_RoA(self, M, rho):
        self.M = M
        self.rho = rho

    def is_in_RoA(self, x):

        x = np.array(x)

        if (x.T @ self.M @ x) < self.rho:
            return True
        else:
            return False
    
    def get_control_output(self, x):
        # Measured State
        meas_pos = x[0] 
        meas_vel = x[1]
        # Desired Goal State
        des_pos = np.pi 
        des_vel = 0.0

        # th = meas_pos + np.pi
        th = meas_pos
        th = (th) % (2*np.pi)

        x_norm = [th, meas_vel]
        err = [th - np.pi, meas_vel]
        
        if self.is_in_RoA(err):
            if self.active_controller != "lqr":
                self.active_controller = "lqr"
                print("Switching to lqr")
            return self.lqr_controller.get_control_output(x_norm)
        else:
            if self.active_controller != "es":
                self.active_controller = "es"
                print("Switching to es")
            return self.energy_shaping_controller.get_control_output(x)
                
        
        # u = self.energy_shaping_controller.get_control_output(x)
        # th = meas_pos + np.pi
        # th = (th + np.pi) % (2*np.pi) - np.pi

        
        # if self.is_in_RoA([th, meas_vel]):
        #     if u is not None:
        #         if self.active_controller != "lqr":
        #             self.active_controller = "lqr"
        #             print("Switching to lqr")
        #         u = self.lqr_controller.get_control_output(x)
        #         #print("lqr: ", u)
        #         #print("u: ",u)
        #     else:
        #         if self.active_controller != "EnergyShaping":
        #             self.active_controller = "EnergyShaping"
        #             print("Switching to EnergyShaping")
        #         u = self.energy_shaping_controller.get_control_output(x)
        #         #print("ES: ", u)
        # #else:
        #     #print("outside RoA")
        # return u