import datetime

import numpy as np
from dynamics import orbit_dynamics
import environment as env
import disturbances as dis


class Simulation:
    def __init__(self, env, disturbances, sensors, estimator,
                 controller, actuators, orbit_dyn, attitude_dyn, logger,
                 dt: datetime.timedelta, t0: datetime.datetime, tf: datetime.datetime):

        self.env = env
        self.disturbances = disturbances
        self.sensors = sensors
        self.estimator = estimator
        self.controller = controller
        self.actuators = actuators
        self.orbit = orbit_dyn
        self.attitude = attitude_dyn
        self.logger = logger

        self.dt = dt
        self.t = t0
        self.tf = tf

        self.last_actuator_output = None

    def run(self):
        while self.t < self.tf:
            
            # current state
            lat, lon, alt = self.orbit.postition_geodedic
            r_eci = self.orbit.state[0:3]
            v_eci = self.orbit.state[3:6]
            q_BI = self.attitude.state[0:4]


            rho = env.atmosphere_density_msis(self.t, lat, lon, alt)
            B = env.magnetic_field_vector(self.t, lat, lon, alt)
            sun_pos = env.sun_position(self.t)
            in_shadow = env.is_in_shadow(r_eci, sun_pos)
            moon_pos = env.moon_position(self.t)
            
            F_third = dis.third_body_forces(r_eci, self.m, sun_pos, moon_pos)
            tau_gg = dis.gravity_gradient(r_eci, v_eci, q_BI, J_B)
            F_aero, tau_aero = dis.aerodynamic_drag(r_eci, v_eci, q_BI, surfaces, rho)
            F_SRP, tau_SRP = dis.solar_radiation_pressure(r_eci, sun_pos, in_shadow, q_BI, surfaces)


            # sensors (no implementation yet)
            sun_pos_mea = sun_pos

            # estimators (no implementation yet)
            sun_pos_est = sun_pos_mea 
            state_est = state

            # controllers
            u_rw = ...
            u_mag = ...
        
            # actuators
            tau_rw, h_rw = ...
            tau_mag = ...


            # integrate dynamics (orbit, attitude, estimators, controllers, actuators)
            d_v_eci = orbit_dynamics(self.m, r, np.zeros(3), F_aero + F_SRP + F_third)
            d_
            

            # logging

            self.t += self.dt
