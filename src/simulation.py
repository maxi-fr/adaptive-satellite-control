import datetime
import json
import os
import csv
from typing import Callable
from scipy.spatial.transform import Rotation as R
import numpy as np
from dynamics import SGP4
import environment as env
import disturbances as dis
from kinematics import euler_ocr_to_sbc, orc_to_eci, orc_to_sbc, quaternion_kinematics, eci_to_geodedic
from satellite import Spacecraft, replace_orientation_matrices, string_to_matrix
from tqdm import tqdm
from estimators import AttitudeEKF
from controllers import simple_rw_mag_controller

class Simulation:
    def __init__(self, sat: Spacecraft,
                 initial_r_ECI: np.ndarray,
                 initial_v_ECI: np.ndarray,
                 initial_attitude_BO: R,
                 initial_ang_vel_B: np.ndarray,
                 log_file: str,
                 dt: datetime.timedelta,
                 t0: datetime.datetime,
                 tf: datetime.datetime):

        self.t0 = t0
        self.dt = dt
        self.tf = tf

        self.sat = sat
        self.inital_state = np.zeros(22)
        # position and velocity in ECI
        self.inital_state[:3] = initial_r_ECI
        self.inital_state[3:6] = initial_v_ECI

        # attitude: quaternion from ECI -> Body
        # R_BI = R_BO * (R_IO)^-1
        self.inital_state[6:10] = (
                initial_attitude_BO * orc_to_eci(self.inital_state[0:3], self.inital_state[3:6]).inv()
                ).as_quat(scalar_first=False)  # quaternion representing rotation from ECI to Body Frame
        
        if initial_ang_vel_B is not None:
            self.inital_state[10:13] = initial_ang_vel_B

        # log_file = log_file[:4]
        # if os.path.exists(log_file):
        #     for i in range(1000):
        #         if not os.path.exists(log_file + "_" + str(i) + ".csv"):
        #             log_file = log_file + "_" + str(i)
        #             break

        self.log_file = log_file

        # with open(self.log_file, 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['t', 'r_eci_x', 'r_eci_y', 'r_eci_z', 'v_eci_x', 'v_eci_y', 'v_eci_z',
        #                     'q_BI_x', 'q_BI_y', 'q_BI_z', 'q_BI_w', 'omega_x', 'omega_y', 'omega_z',
        #                      'omega_rw', 'tau_rw', 'i_mag'])

        # Attitude EKF
        q0 = self.inital_state[6:10]          # initial quaternion
        b0 = np.zeros(3)                      # zero gyro bias
        P0 = np.eye(6) * 1e-6                 # initial covariance error
        Qc = np.diag([
            1e-8, 1e-8, 1e-8,                 # process noise for attitude
            1e-10, 1e-10, 1e-10               # process noise for bias
        ])
        R_sun = np.eye(3) * (np.deg2rad(0.5) ** 2)   # around 0.5° noise
        R_mag = np.eye(3) * (1e-7 ** 2)              # just a guess

        self.att_ekf = AttitudeEKF(q0, b0, P0, Qc, R_sun, R_mag)
        self.q_ref = q0.copy()                # we want to keep the initial attitude
        self.omega_ref = np.zeros(3)          # reference angular velocity = 0

            
    @classmethod
    def from_json(cls, eos_file_path: str) -> "Simulation":
        with open(eos_file_path, "r") as f:
            eos_file = json.load(f)

        data: dict = replace_orientation_matrices(eos_file)  # type: ignore

        settings = data["Settings"] # type: ignore
        t0 = datetime.datetime.fromisoformat(settings["SimulationStart"]+"Z")

        dt_ = tuple(map(float, settings["SimulationPeriod"].split(":")))
        dt = datetime.timedelta(hours=dt_[0], minutes=dt_[1], seconds=dt_[2])

        dur_ = tuple(map(int, settings["SimulationDuration"].split(":")))
        dur = datetime.timedelta(hours=dur_[0], minutes=dur_[1], seconds=dur_[2])
        tf = t0 + dur

        kin_model = data["ModelObjects"]["Kinematic Model"] # type: ignore
        roll = kin_model["InitialRoll"] # deg
        pitch = kin_model["InitialPitch"]
        yaw = kin_model["InitialYaw"]
        init_ang_vel_B_BI = string_to_matrix(kin_model["InitialRates"])
        init_att_BO = euler_ocr_to_sbc(roll, pitch, yaw)

        tle1 = data["ModelObjects"]["Orbit Model"]["Tle1"] # type: ignore
        tle2 = data["ModelObjects"]["Orbit Model"]["Tle2"] # type: ignore
        orbit_model = SGP4.from_tle(tle1, tle2)
        r_ECI, v_ECI = orbit_model.propagate(t0)

        return cls(Spacecraft.from_eos_file(data, dt), r_ECI, v_ECI, init_att_BO, init_ang_vel_B_BI, "Log_file.csv", dt, t0, tf)

    def run(self):
        state = self.inital_state
        t = self.t0
        u = np.zeros(6)
        
        total_steps = int((self.tf - self.t0).total_seconds() / self.dt.total_seconds())
        with tqdm(total=total_steps, desc="Running Simulation") as pbar:
            while t < self.tf:
                # Necessary for sensor to get current measurements. k1 is passed to integration step, to avoid recomputation
                k1 = self.world_dynamics(state, u, t, update_sensors=True)
                # Flight software (FSW)
                sun_mea, _ = self.sat.sun_sensor.read(t)
                mag_mea, _ = self.sat.magnetometer.read(t)
                gps_mea, _ = self.sat.gps.read(t)
                acc_mea, _ = self.sat.accelerometer.read(t)
                gyro_mea, _ = self.sat.gyro.read(t)
                omega_rw_mea = np.array([rw.read(t)[0] for rw in self.sat.rw_speed_sensors])

                # we need r_eci, v_eci for logic/estimator
                r_eci = state[0:3]
                v_eci = state[3:6]

                # AttitudeEKF
                # 1) prediction from Gyro
                self.att_ekf.predict(t, gyro_mea)

                # 2) update from Sun + Magnometer
                self.att_ekf.update_sun(t, sun_mea)
                self.att_ekf.update_mag(t, mag_mea, r_eci)

                # estimated state from EKF
                q_est, b_est, P_est = self.att_ekf.get_state()
                omega_est = gyro_mea - b_est  # gyro without bias
                state_est = np.hstack([r_eci, v_eci, q_est, omega_est])

                # Sun position estimate (z env modelu, nie zo senzora)
                sun_pos_est = env.sun_position(t)

                # TODO: controllers
                q_ref = self.q_ref
                omega_ref = self.omega_ref

                # abritrary gains (tuning later)
                Kp_q = 1e-3
                Kd_w = 1e-2

                # torque constant of RW (all 3 RW's are the same)
                K_t_rw = self.sat.rws[0].K_t

                u_rw, u_mag = simple_rw_mag_controller(
                    q_est=q_est,
                    omega_est=omega_est,
                    q_ref=q_ref,
                    omega_ref=omega_ref,
                    Kp_q=Kp_q,
                    Kd_w=Kd_w,
                    K_t_rw=K_t_rw,
                )

                max_i = self.sat.rws[0].max_current
                u_rw = np.clip(u_rw, -max_i, max_i)

                # so far, I don't want to saturate MTQ (they're around 0 anyway)
                u_mag = np.asarray(u_mag, dtype=float).reshape(3)
                # logging 
                # TODO: maybe log to different files. 
                # Because it could get alot with all different variables: states, measured states, estimated states, environment variables
                with open(self.log_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([t] + list(state))

                # integrate world dynamics
                u = np.hstack((u_mag, u_rw))
                next_state = rk4_step(self.world_dynamics, state, u, t, self.dt, k1)

                t += self.dt
                state = next_state
                pbar.update(1)

    def world_dynamics(self, x: np.ndarray, u: np.ndarray, t: datetime.datetime, update_sensors: bool = False):
        """
        Helper function to wrap dynamics for whole system for integration.

        Parameters
        ----------
        x : np.ndarray, shape (22,)
            All variable states during integration
        u : np.ndarray, shape (6,)
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

        #TODO: rotations B->O and B->I get recomputed many times. Speed up by handling them better
        F_grav = dis.non_spherical_gravity_forces(r_eci, self.sat.m)
        F_third = dis.third_body_forces(r_eci, self.sat.m, sun_pos, moon_pos)
        tau_gg = dis.gravity_gradient(r_eci, v_eci, q_BI, self.sat.J_B)
        F_aero, tau_aero = dis.aerodynamic_drag(r_eci, v_eci, q_BI, self.sat.surfaces, rho)
        F_SRP, tau_SRP = dis.solar_radiation_pressure(r_eci, sun_pos, in_shadow, q_BI, self.sat.surfaces)

        # reaction wheel torque and internal angular momentum (sum over all wheels)
        tau_rw = np.zeros(3)
        h_rw = np.zeros(3)
        for i, rw in enumerate(self.sat.rws):
            tau_i, h_i = rw.torque_ang_momentum(rws_curr[i], omega_rws[i], omega)
            tau_rw += tau_i
            h_rw += h_i

        # magnetic torques from all magnetorquers
        tau_mag = np.zeros(3)
        for i, mag in enumerate(self.sat.mag):
            tau_mag += mag.torque(mag_curr[i], B)

        # differential equations
        d_r = v_eci
        d_v = self.sat.orbit_dynamics(r_eci, np.zeros(3), F_aero + F_SRP + F_third + F_grav)
        d_q = quaternion_kinematics(q_BI, omega)
        d_omega = self.sat.attitude_dynamics(omega, h_rw, tau_mag - tau_rw, tau_gg + tau_aero + tau_SRP)
        # reaction wheel speed and current dynamics (one state per wheel)
        d_omega_rw = np.zeros(3)
        d_curr_rw = np.zeros(3)
        for i, rw in enumerate(self.sat.rws):
            d_omega_rw[i], d_curr_rw[i] = rw.dynamics(
                u=u_rw[i],
                omega_dot_body=d_omega,
                i=rws_curr[i],
            )
        # magnetorquer current dynamics (one state per coil)
        d_curr_mag = np.zeros(3)
        for i, mag in enumerate(self.sat.mag):
            d_curr_mag[i] = mag.dynamics(
                u=u_mag[i],
                i=mag_curr[i],
            )

        dx = np.hstack((d_r, d_v, d_q, d_omega, d_omega_rw, d_curr_rw, d_curr_mag))

        if update_sensors:
            self.sat.sun_sensor.measure(t, sun_pos)
            self.sat.magnetometer.measure(t, B)
            self.sat.gps.measure(t, r_eci)
            self.sat.accelerometer.measure(t, d_v, orc_to_sbc(q_BI, r_eci, v_eci))
            self.sat.gyro.measure(t, omega)
            for i, rw in enumerate(self.sat.rw_speed_sensors):
                rw.measure(t, omega_rws[i])

        return dx

# TODO: maybe implement a variable step size integrator. RK45 (EOS uses simple rk4)
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
    k1 : np.ndarray|None = None
        First grid point at current time. If None then it is computed from f.

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
    k4 = f(x + dt_float * k3, u, t + dt)
    return x + (dt_float/ 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)