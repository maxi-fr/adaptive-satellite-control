

import json
from typing import List
import numpy as np
from actuators import Magnetorquer, ReactionWheel
from dynamics import orbit_dynamics, attitude_dynamics
from sensors import SunSensor, Magnetometer, GPS, Accelerometer, Gyroscope, RW_tachometer
from utils import Surface, string_to_matrix


"""
The idea of these classes is that they hold the parameters and provide wrappers for the dynamics functions 
but they do not contain the state. The state is managed externally.
(TODO: maybe doesnt have to be a wrapper, they can just implement the dynamics) 
"""

class Spacecraft:

    def __init__(self, m, J_B, surfaces: list[Surface], rws: list[ReactionWheel], magnetorquers: list[Magnetorquer],
                 sun_sensor: SunSensor, magnetometer: Magnetometer, gps: GPS, accelerometer: Accelerometer, gyro: Gyroscope,
                 rw_speed_sensors: list[RW_tachometer]) -> None:
        self.m = m
        self.J_B = J_B

        self.sun_sensor = sun_sensor
        self.magnetometer = magnetometer
        self.gps = gps
        self.accelerometer = accelerometer
        self.gyro = gyro
        self.rw_speed_sensors = rw_speed_sensors

        self.rws = rws
        self.mag = magnetorquers

        J_rw = sum([rws[i].inertia * rws[i].spin_axis @ rws[i].spin_axis.T 
                       for i in range(len(rws))])
        
        self.J_tilde = self.J_B - J_rw
        self.surfaces = surfaces

    @classmethod
    def from_eos_file(cls, data: dict, dt):
        #TODO: correctly initialize sensors and actuators
        
        trace = data["ModelObjects"]["TRACE"]

        m = trace["StructureMass"]
        J_B = string_to_matrix(trace["StructureMomentOfInertia"])

        surfaces = [Surface.from_eos_panel(v) for k, v in data["ModelObjects"].items() if "panel" in k.lower()]
        rws = [ReactionWheel(0.2e-3, 6000, 2e-6, a) for a in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        magnetorquers = [Magnetorquer(0.2e-2, a) for a in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]

        dt = 0.01
        sun_sensor = SunSensor(dt, 0.0)
        magnetometer = Magnetometer(dt, 0.0, np.array([0, 0, 0]))
        gps = GPS(dt, 0.0)
        accelerometer = Accelerometer(dt, 0.0, 0.0)
        gyro = Gyroscope(dt, 0.0, 0.0)
        rw_speed_sensors = [RW_tachometer(dt, 0.0) for _ in rws]

        return cls(m, J_B, surfaces, rws, magnetorquers, sun_sensor, magnetometer, gps, accelerometer, gyro, rw_speed_sensors)
        

    def orbit_dynamics(self, r_eci: np.ndarray, ctrl_force: np.ndarray, dist_force: np.ndarray):

        return orbit_dynamics(self.m, r_eci, ctrl_force, dist_force)

    def attitude_dynamics(self, omega: np.ndarray, h_int: np.ndarray, ctrl_torque: np.ndarray, dist_torque: np.ndarray):

        return attitude_dynamics(omega, self.J_tilde, ctrl_torque, dist_torque, h_int)

