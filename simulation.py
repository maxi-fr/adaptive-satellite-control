import datetime
import os
import csv
from typing import Callable
from scipy.spatial.transform import Rotation as R
import numpy as np
from dynamics import KeplarElements
import environment as env
import disturbances as dis
from kinematics import orc_to_eci, quaternion_kinematics, eci_to_geodedic
from satellite import Spacecraft


class Simulation:
    def __init__(self, sat: Spacecraft, keplar_elements: KeplarElements, initial_attitude: R, initial_ang_vel_B: np.ndarray, log_file: str,
                 dt: datetime.timedelta, t0: datetime.datetime, tf: datetime.datetime):

        self.t0 = t0
        self.dt = dt
        self.tf = tf

        self.sat = sat
        self.inital_state = np.zeros(13)
        self.inital_state[:6] = keplar_elements.to_eci()

        # BI = BO * (IO)-1
        self.inital_state[6:10] = (initial_attitude * orc_to_eci(self.inital_state[0:3], self.inital_state[3:6]).inv()
                            ).as_quat(scalar_first=False)  # quaternion representing rotation from ECI to Body Frame
        if initial_ang_vel_B is not None:
            self.inital_state[10:13] = initial_ang_vel_B

        if os.path.exists(log_file):
            for i in range(1000):
                if not os.path.exists(log_file + "_" + str(i)):
                    log_file = log_file + "_" + str(i)
                    break

        self.log_file = log_file

        with open(self.log_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'r_eci_x', 'r_eci_y', 'r_eci_z', 'v_eci_x', 'v_eci_y', 'v_eci_z',
                            'q_BI_x', 'q_BI_y', 'q_BI_z', 'q_BI_w', 'omega_x', 'omega_y', 'omega_z',
                             'omega_rw', 'tau_rw', 'i_mag'])

    def run(self):
        state = self.inital_state
        t = self.t0
        u = np.zeros(6)

        while t < self.tf:
            
            # Flight software (FSW)
            # TODO: sensors 
            k1 = self.world_dynamics(state, u, t, update_sensors=True)
            sun_pos_mea = self.sat.sun_sensor.read()
            moon_pos_mea = self.sat.moon_sensor.read()
            self.sat.magnetometer.read()
            self.sat.gps.read()
            self.sat.imu.read()
            
            # TODO: estimators
            sun_pos_est = sun_pos_mea
            state_est = state

            # TODO: controllers
            u_rw = np.zeros(3)
            u_mag = np.zeros(3)
            
            # logging
            with open(self.log_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([t] + list(state))
 
            # integrate world dynamics (envrionment, orbit, attitude, sensors, actuators)
            u = np.vstack((u_rw, u_mag))
            next_state = rk4_step(self.world_dynamics, state, u, t, self.dt, k1)

            t += self.dt
            state = next_state

    def world_dynamics(self, x: np.ndarray, u: np.ndarray, t: datetime.datetime, update_sensors: bool = False):
        """
        Helper function to wrap dynamics for whole system for integration.

        Parameters
        ----------
        x : np.ndarray, shape (22,)
            All variable states during integration
        u : np.ndarray, shape ()
            All control inputs during integration. They are constant during integration.
        t : datetime.datetime
            Current simulation time.
        update_sensors : bool = False
            Flag whether the sensor should be updates with new measurements. Default is False.

        Returns
        -------
        np.ndarray, shape (22,)
            State derivative dx/dt at time t.
        """

        r_eci = x[0:3]
        v_eci = x[3:6]
        q_BI = x[6:10]
        omega = x[10:13]
        omega_rws = x[13:16]
        rws_curr = x[16:19]
        mag_curr = x[19:22]
        
        u_mag = u[:3]
        u_rw = u[3:6]

        lat, lon, alt = eci_to_geodedic(r_eci)

        # algebraic relations
        rho = env.atmosphere_density_msis(t, lat, lon, alt)
        B = env.magnetic_field_vector(t, lat, lon, alt)
        sun_pos = env.sun_position(t)
        in_shadow = env.is_in_shadow(r_eci, sun_pos)
        moon_pos = env.moon_position(t)

        F_third = dis.third_body_forces(r_eci, self.sat.m, sun_pos, moon_pos)
        tau_gg = dis.gravity_gradient(r_eci, v_eci, q_BI, self.sat.J_B)
        F_aero, tau_aero = dis.aerodynamic_drag(r_eci, v_eci, q_BI, self.sat.surfaces, rho)
        F_SRP, tau_SRP = dis.solar_radiation_pressure(r_eci, sun_pos, in_shadow, q_BI, self.sat.surfaces)

        tau_rw, h_rw = sum([np.array(rw.torque_ang_momentum(rws_curr[i], omega_rws[i])) for i, rw in enumerate(self.sat.rws)])
        tau_mag = sum([np.array(mag.torque(mag_curr[i], B)) for i, mag in enumerate(self.sat.mag)])

        # differential equations
        d_r = v_eci
        d_v = self.sat.orbit_dynamics(r_eci, np.zeros(3), F_aero + F_SRP + F_third)
        d_q = quaternion_kinematics(q_BI, omega)
        d_omega = self.sat.attitude_dynamics(omega, h_rw, tau_mag - tau_rw, tau_gg + tau_aero + tau_SRP)
        d_omega_rw, d_curr_rw = np.array([rw.dynamics(u_rw[i], d_omega, rws_curr[i]) for i, rw in enumerate(self.sat.rws)])
        d_curr_mag = np.array([mag.dynamics(u_mag[i], mag_curr[i]) for i, mag in enumerate(self.sat.mag)]) 

        dx = np.vstack((d_r, d_v, d_q, d_omega, d_omega_rw, d_curr_rw, d_curr_mag))

        # sensors
        if update_sensors:
            self.sat.sun_sensor.measure(t, sun_pos)
            self.sat.moon_sensor.measure(t, moon_pos)
            self.sat.magnetometer.measure(t, B)
            self.sat.gps.measure(t, r_eci)
            self.sat.imu.measure(t, d_v, d_omega)

        return dx


def rk4_step(f: Callable[[np.ndarray, np.ndarray, datetime.datetime], np.ndarray], 
             x: np.ndarray, u: np.ndarray, t: datetime.datetime, dt: datetime.timedelta, k1: np.ndarray|None = None) -> np.ndarray:
    """
    Classic 4th-order Runge-Kutta integrator.

    Parameters
    ----------
    f : Callable[[np.ndarray, np.ndarray, datetime.datetime], np.ndarray]
        Function f(x, u, t) -> dx/dt that computes the state derivative.
    x : np.ndarray
        Current state.
    u : np.ndarray
        Current input
    t : datetime.datetime
        Current simulation time.
    dt : float
        Time step.
    k1 : np.ndarray|None
        First grid point at current time. If none then it is computed from f.

    Returns
    -------
    np.ndarray
        State after one time step.
    """

    dt_float = dt.total_seconds()

    if k1 is None:
        k1 = f(x, u, t)

    k2 = f(x + 0.5 * dt_float * k1, u, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt_float * k2, u, t + 0.5 * dt)
    k4 = f(x + dt * k3, u, t + dt)
    return x + (dt_float/ 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)