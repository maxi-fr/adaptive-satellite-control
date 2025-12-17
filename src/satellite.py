

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
    """
    A class representing a satellite, holding its physical parameters, sensors, and actuators.

    This class acts as a container for the satellite's configuration and provides wrappers
    for dynamics functions using the satellite's physical properties.
    """

    def __init__(self, m, J_B, surfaces: list[Surface], rws: list[ReactionWheel], magnetorquers: list[Magnetorquer],
                 sun_sensor: SunSensor, magnetometer: Magnetometer, gps: GPS, accelerometer: Accelerometer, gyro: Gyroscope,
                 rw_speed_sensors: list[RW_tachometer]) -> None:
        """
        Initialize the Spacecraft object.

        Parameters
        ----------
        m : float
            Mass of the satellite [kg].
        J_B : np.ndarray
            Inertia tensor of the satellite in the body frame [kg*m^2].
        surfaces : list[Surface]
            List of surface elements defining the satellite geometry.
        rws : list[ReactionWheel]
            List of reaction wheel actuators.
        magnetorquers : list[Magnetorquer]
            List of magnetorquer actuators.
        sun_sensor : SunSensor
            Sun sensor object.
        magnetometer : Magnetometer
            Magnetometer object.
        gps : GPS
            GPS receiver object.
        accelerometer : Accelerometer
            Accelerometer object.
        gyro : Gyroscope
            Gyroscope object.
        rw_speed_sensors : list[RW_tachometer]
            List of reaction wheel tachometers.
        """
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
        """
        Creates a Spacecraft instance from a parsed EOS JSON configuration.

        Parameters
        ----------
        data : dict
            Dictionary containing the parsed EOS model data.
        dt : float
            Simulation time step [s].

        Returns
        -------
        Spacecraft
            An initialized Spacecraft instance.
        """
        #TODO: correctly initialize sensors and actuators
        
        trace = data["ModelObjects"]["TRACE"]

        m = trace["StructureMass"]
        J_B = string_to_matrix(trace["StructureMomentOfInertia"])

        surfaces = [Surface.from_eos_panel(v) for k, v in data["ModelObjects"].items() if "panel" in k.lower()]
        # Reaction Wheels
        RW_MAX_TORQUE = 2e-3  # [N·m]
        RW_MAX_RPM = 6000  # [rpm]
        RW_INERTIA = 2.82e-6  # [kg·m^2]
        rws = [
            ReactionWheel(RW_MAX_TORQUE, RW_MAX_RPM, RW_INERTIA, np.array([-1, 0, 0])),  # -X
            ReactionWheel(RW_MAX_TORQUE, RW_MAX_RPM, RW_INERTIA, np.array([0, 1, 0])),  # +Y
            ReactionWheel(RW_MAX_TORQUE, RW_MAX_RPM, RW_INERTIA, np.array([0, 0, 1])),  # +Z
        ]
        # Magnetorquers
        MTQ_XY = 0.3  # [A·m^2]  CR0003
        MTQ_Z = 0.2  # [A·m^2]  CR0002
        magnetorquers = [
            Magnetorquer(MTQ_XY, np.array([-1, 0, 0])),  # -X
            Magnetorquer(MTQ_XY, np.array([0, 1, 0])),  # +Y
            Magnetorquer(MTQ_Z, np.array([0, 0, 1])),  # +Z
        ]
        # Sensor sampling frequencies [Hz]
        GYRO_FREQUENCY = 10.0
        MAG_FREQUENCY = 2.0
        SUN_FREQUENCY = 1.0
        GPS_FREQUENCY = 0.2
        ACC_FREQUENCY = 10.0
        RW_TACHO_FREQ = 5.0

        # Sensors
        sun_sensor = SunSensor(SUN_FREQUENCY, sigma_sq=0.0)
        magnetometer = Magnetometer(MAG_FREQUENCY, sigma_sq=0.0, const_bias=np.zeros(3))
        gps = GPS(GPS_FREQUENCY, sigma_sq=0.0)
        accelerometer = Accelerometer(ACC_FREQUENCY, sigma_sq=0.0, bias_sigma_sq=0.0)
        gyro = Gyroscope(GYRO_FREQUENCY, sigma_sq=0.0, bias_sigma_sq=0.0)
        rw_speed_sensors = [RW_tachometer(RW_TACHO_FREQ, sigma_sq=0.0) for _ in rws]

        return cls(m, J_B, surfaces, rws, magnetorquers, sun_sensor, magnetometer, gps, accelerometer, gyro, rw_speed_sensors)
        

    def orbit_dynamics(self, r_eci: np.ndarray, ctrl_force: np.ndarray, dist_force: np.ndarray):
        """
        Computes the orbital dynamics (acceleration) of the spacecraft.

        Parameters
        ----------
        r_eci : np.ndarray
            Position vector in the ECI frame [m].
        ctrl_force : np.ndarray
            Control force vector in the ECI frame [N].
        dist_force : np.ndarray
            Disturbance force vector in the ECI frame [N].

        Returns
        -------
        np.ndarray
            Acceleration vector in the ECI frame [m/s^2].
        """
        return orbit_dynamics(self.m, r_eci, ctrl_force, dist_force)

    def attitude_dynamics(self, omega: np.ndarray, h_int: np.ndarray, ctrl_torque: np.ndarray, dist_torque: np.ndarray):
        """
        Computes the attitude dynamics (angular acceleration) of the spacecraft.

        Parameters
        ----------
        omega : np.ndarray
            Angular velocity vector in the body frame [rad/s].
        h_int : np.ndarray
            Total internal angular momentum vector in the body frame [N*m*s].
        ctrl_torque : np.ndarray
            Control torque vector in the body frame [N*m].
        dist_torque : np.ndarray
            Disturbance torque vector in the body frame [N*m].

        Returns
        -------
        np.ndarray
            Angular acceleration vector in the body frame [rad/s^2].
        """
        return attitude_dynamics(omega, self.J_tilde, ctrl_torque, dist_torque, h_int)
