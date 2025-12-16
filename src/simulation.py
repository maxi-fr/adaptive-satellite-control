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
from kinematics import eci_to_sbc, euler_ocr_to_sbc, orc_to_eci, orc_to_sbc, quaternion_kinematics, eci_to_geodedic
from satellite import Spacecraft, string_to_matrix
from tqdm import tqdm

from utils import Logger, PiecewiseConstant, floor_time_to_minute, replace_orientation_matrices


class Simulation:
    def __init__(self, sat: Spacecraft, initial_r_ECI: np.ndarray, initial_v_ECI: np.ndarray, initial_attitude_BO: R, initial_ang_vel_B: np.ndarray,
                 dt: datetime.timedelta, t0: datetime.datetime, tf: datetime.datetime):

        self.t0 = t0
        self.dt = dt
        self.tf = tf

        self.sat = sat
        self.inital_state = np.zeros(22)
        self.inital_state[:3] = initial_r_ECI
        self.inital_state[3:6] = initial_v_ECI

        # R_BI = R_BO * (R_IO)^-1
        self.inital_state[6:10] = (initial_attitude_BO * orc_to_eci(self.inital_state[0:3],
                                                                    self.inital_state[3:6]).inv()
                                  ).as_quat(scalar_first=False)  # quaternion representing rotation from ECI to Body Frame
        
        if initial_ang_vel_B is not None:
            self.inital_state[10:13] = initial_ang_vel_B

        # ===== info log =====
        self.log_folder = "Simulation_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.state_logger = Logger(os.path.join(self.log_folder, "state.csv"), 
                        ['t', 'r_eci_x', 'r_eci_y', 'r_eci_z', 'v_eci_x', 'v_eci_y', 'v_eci_z',
                            'q_BI_x', 'q_BI_y', 'q_BI_z', 'q_BI_w', 'omega_x', 'omega_y', 'omega_z',
                             'omega_rw_1', 'omega_rw_2', 'omega_rw_3',
                             'i_rw_1', 'i_rw_2', 'i_rw_3', 'i_mag_1', 'i_mag_2', 'i_mag_3'])
        self.input_logger = Logger(os.path.join(self.log_folder, "input.csv"), 
                    ['t', 'u_mag_1', 'u_mag_2', 'u_mag_3', 'u_rw_1', 'u_rw_2', 'u_rw_3'])
        self.quat_logger = Logger(
            os.path.join(self.log_folder, "quat.csv"),
            ['t', 'q_BI_x', 'q_BI_y', 'q_BI_z', 'q_BI_w', 'q_norm', 'omega_norm']
        )

        # Sun and moon position change very slowly. For performance a new value is only calculated every minute
        self.sun_position = PiecewiseConstant(fn=env.sun_position, time_bucket_fn=floor_time_to_minute)
        self.moon_position = PiecewiseConstant(fn=env.moon_position, time_bucket_fn=floor_time_to_minute)

            
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

        return cls(Spacecraft.from_eos_file(data, dt), r_ECI, v_ECI, init_att_BO, init_ang_vel_B_BI, dt, t0, tf)

    def run(self):
        state = self.inital_state
        t = self.t0
        u = np.zeros(6)

        with tqdm(total=(self.tf - self.t0).total_seconds()/60, desc="Simulation time", unit="sim min") as pbar:
            while t < self.tf:
    
                k1 = self.world_dynamics(state, u, t, update_sensors=True) # Necessary for sensor get get current measurements. k1 is passed to integration step, to avoid recomputation 
                
                # Flight software (FSW)
                sun_mea = self.sat.sun_sensor.read(t)
                mag_mea =self.sat.magnetometer.read(t)
                gps_mea = self.sat.gps.read(t)
                acc_mea = self.sat.accelerometer.read(t)
                gyro_mea = self.sat.gyro.read(t)
                omega_rw_mea = [rw.read(t) for rw in self.sat.rw_speed_sensors]
    
                # TODO: estimators
                sun_pos_est = sun_mea
                state_est = state
    
                # TODO: controllers
                u_mag = np.zeros(3)
                u_rw = np.zeros(3)

                # integrate world dynamics
                u = np.concatenate((u_mag, u_rw))
                next_state = rk4_step(self.world_dynamics, state, u, t, self.dt, k1)

                # quaternion normalization (x,y,z,w)
                q = next_state[6:10]
                qn = np.linalg.norm(q)
                if qn == 0:
                    raise ValueError("Quaternion norm became zero.")
                q = q / qn
                next_state[6:10] = q

                # normalizing omega
                omega_norm = float(np.linalg.norm(next_state[10:13]))
                # logging (after normalization)
                self.state_logger.log([t + self.dt] + list(next_state))
                self.input_logger.log([t + self.dt] + list(u))
                self.quat_logger.log([t + self.dt,
                                      q[0], q[1], q[2], q[3],
                                      float(np.linalg.norm(q)),
                                      omega_norm])

                t += self.dt
                state = next_state
                pbar.update(self.dt.total_seconds()/60)
        

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

        R_BO = orc_to_sbc(q_BI, r_eci, v_eci)
        R_BI = eci_to_sbc(q_BI)
        lat, lon, alt = eci_to_geodedic(r_eci)

        # algebraic relations
        rho = env.atmosphere_density_msis(t, lat, lon, alt)
        B = env.magnetic_field_vector(t, lat, lon, alt)
        sun_pos: np.ndarray = self.sun_position(t) #type: ignore
        in_shadow = env.is_in_shadow(r_eci, sun_pos)
        moon_pos: np.ndarray = self.moon_position(t) #type: ignore

        F_grav = dis.non_spherical_gravity_forces(r_eci, self.sat.m)
        F_third = dis.third_body_forces(r_eci, self.sat.m, sun_pos, moon_pos)
        tau_gg = dis.gravity_gradient(r_eci, R_BO, self.sat.J_B)
        F_aero, tau_aero = dis.aerodynamic_drag(r_eci, v_eci, R_BI, self.sat.surfaces, rho)
        F_SRP, tau_SRP = dis.solar_radiation_pressure(r_eci, sun_pos, in_shadow, R_BI, self.sat.surfaces)

        tau_rw, h_rw = sum([np.array(rw.torque_ang_momentum(rws_curr[i], omega_rws[i], omega)) for i, rw in enumerate(self.sat.rws)]) # type: ignore
        tau_mag = sum([np.array(mag.torque(mag_curr[i], B)) for i, mag in enumerate(self.sat.mag)])

        # differential equations
        d_r = v_eci
        d_v = self.sat.orbit_dynamics(r_eci, np.zeros(3), F_aero + F_SRP + F_third + F_grav)
        d_q = quaternion_kinematics(q_BI, omega)
        d_omega = self.sat.attitude_dynamics(omega, h_rw, tau_mag - tau_rw, tau_gg + tau_aero + tau_SRP)
        d_omega_rw, d_curr_rw = np.array([rw.dynamics(u_rw[i], d_omega, rws_curr[i]) for i, rw in enumerate(self.sat.rws)]).T
        d_curr_mag = np.array([mag.dynamics(u_mag[i], mag_curr[i]) for i, mag in enumerate(self.sat.mag)]) 

        dx = np.concatenate((d_r, d_v, d_q, d_omega, d_omega_rw, d_curr_rw, d_curr_mag))

        if update_sensors:
            self.sat.sun_sensor.measure(t, sun_pos)
            self.sat.magnetometer.measure(t, B)
            self.sat.gps.measure(t, r_eci)
            self.sat.accelerometer.measure(t, d_v, orc_to_sbc(q_BI, r_eci, v_eci))
            self.sat.gyro.measure(t, omega)
            for i, rw in enumerate(self.sat.rw_speed_sensors):
                rw.measure(t, omega_rws[i])

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


if __name__ == "__main__":
    eos_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tudsat-trace_eos.json")
    
    sim = Simulation.from_json(eos_file_path)

    sim.tf = sim.t0 + datetime.timedelta(hours=1)
    sim.dt = datetime.timedelta(seconds=1)

    sim.run()