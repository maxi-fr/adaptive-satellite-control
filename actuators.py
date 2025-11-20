import numpy as np


class ReactionWheel:
    def __init__(self, max_torque: float, max_rpm: float, inertia: float, spin_axis: np.ndarray):
        self.max_torque = max_torque
        self.max_rpm = max_rpm
        self.inertia = inertia
        self.spin_axis = spin_axis
        self.current_rpm = 0.0

