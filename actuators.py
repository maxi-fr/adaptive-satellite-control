import numpy as np

class ReactionWheel:    
    def __init__(self, max_torque: float, max_rpm: float, inertia: float, spin_axis: np.ndarray):
        self.max_torque = max_torque
        self.max_rpm = max_rpm
        self.inertia = inertia
        self.spin_axis = spin_axis


    def torque_ang_momentum(self, i, omega_w, omega) -> tuple[float, float]:
        # TODO
        h_wheel = (self.inertia * (omega_w + np.dot(self.spin_axis, omega))) * self.spin_axis  
        tau = self.K_t * i
        return tau, h_wheel
    
    def dynamics(self):
        # TODO
        pass



class Magnetorquer:
    def __init__(self, max_moment: float, spin_axis: np.ndarray):
        self.max_moment = max_moment
        self.spin_axis = spin_axis

    def torque(self, i, B) -> float:
        # TODO
        tau = 0
        return tau
    
    def dynamics(self):
        # TODO
        pass