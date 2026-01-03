

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
                 sun_sensor: SunSensor, magnetometer: Magnetometer, gps: GPS, gyro: Gyroscope,
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
        self.gyro = gyro
        self.rw_speed_sensors = rw_speed_sensors

        self.rws = rws
        self.mag = magnetorquers

        J_rw = sum([rws[i].inertia * rws[i].axis @ rws[i].axis.T 
                       for i in range(len(rws))])
        
        self.J_tilde = self.J_B - J_rw
        self.surfaces = surfaces


    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a Spacecraft instance from a dictionary.
        """
        m = data["Mass"]
        J_B = np.array(data["Inertia"])

        surfaces = [Surface.from_dict(n, s) for n, s in data["Surfaces"].items()]

        rws = []
        if "Actuators" in data and "ReactionWheels" in data["Actuators"]:
            for rw in data["Actuators"]["ReactionWheels"]:
                rws.append(ReactionWheel(**rw))

        magnetorquers = []
        if "Actuators" in data and "Magnetorquers" in data["Actuators"]:
            for mtq in data["Actuators"]["Magnetorquers"]:
                magnetorquers.append(Magnetorquer(**mtq))

        sensors = data.get("Sensors", {})

        sun_conf = sensors.get("SunSensor", {})
        sun_sensor = SunSensor(sun_conf.get("frequency", 1.0), sigma_sq=sun_conf.get("sigma_sq", 0.0))

        mag_conf = sensors.get("Magnetometer", {})
        magnetometer = Magnetometer(mag_conf.get("frequency", 2.0), sigma_sq=mag_conf.get("sigma_sq", 0.0),
                                    const_bias=np.array(mag_conf.get("const_bias", [0, 0, 0])))

        gps_conf = sensors.get("GPS", {})
        gps = GPS(gps_conf.get("frequency", 0.2), sigma_sq=gps_conf.get("sigma_sq", 0.0))

        gyro_conf = sensors.get("Gyroscope", {})
        gyro = Gyroscope(gyro_conf.get("frequency", 10.0), sigma_sq=gyro_conf.get("sigma_sq", 0.0),
                         bias_sigma_sq=gyro_conf.get("bias_sigma_sq", 0.0))

        rw_speed_sensors = []
        if "RW_tachometers" in sensors:
            for tacho_conf in sensors["RW_tachometers"]:
                rw_speed_sensors.append(RW_tachometer(tacho_conf.get("frequency", 5.0), sigma_sq=tacho_conf.get("sigma_sq", 0.0)))

        return cls(m, J_B, surfaces, rws, magnetorquers, sun_sensor, magnetometer, gps, gyro, rw_speed_sensors)

    def to_dict(self):
        """
        Converts the Spacecraft instance to a dictionary.
        """
        data = {
            "Mass": self.m,
            "Inertia": self.J_B.tolist(),
            "Surfaces": {s.name: s.to_dict() for s in self.surfaces},
            "Sensors": {},
            "Actuators": {}
        }

        if self.sun_sensor is not None:
            data["Sensors"]["SunSensor"] = self.sun_sensor.to_dict()

        if self.magnetometer is not None:
            data["Sensors"]["Magnetometer"] = self.magnetometer.to_dict()
        if self.gps is not None:
            data["Sensors"]["GPS"] = self.gps.to_dict()

        if self.gyro is not None:
            data["Sensors"]["Gyroscope"] = self.gyro.to_dict()

        if self.rw_speed_sensors:
            data["Sensors"]["RW_tachometers"] = [rw.to_dict() for rw in self.rw_speed_sensors]

        if self.rws:
            data["Actuators"]["ReactionWheels"] = [rw.to_dict() for rw in self.rws]

        if self.mag:
            data["Actuators"]["Magnetorquers"] = [mag.to_dict() for mag in self.mag]
        
        return data

    def orbit_dynamics(self, r_eci: np.ndarray, ext_force: np.ndarray):
        """
        Computes the orbital dynamics (acceleration) of the spacecraft.

        Parameters
        ----------
        r_eci : np.ndarray
            Position vector in the ECI frame [m].
        ext_force : np.ndarray
            External force vector in the ECI frame [N].

        Returns
        -------
        np.ndarray
            Acceleration vector in the ECI frame [m/s^2].
        """
        return orbit_dynamics(self.m, r_eci, ext_force)

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
