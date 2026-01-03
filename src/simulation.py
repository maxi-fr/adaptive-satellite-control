import datetime
import json
import os
from typing import Callable
from scipy.spatial.transform import Rotation as R
import warnings
import numpy as np

from actuators import to_current_commands
import controllers
from controllers import Controller
from dynamics import SGP4
import environment as env
import disturbances as dis
from estimators import AttitudeEKF
from kinematics import eci_to_sbc, euler_ocr_to_sbc, orc_to_eci, orc_to_sbc, quaternion_kinematics, eci_to_geodedic
from satellite import Spacecraft, string_to_matrix
from tqdm import tqdm

from utils import Logger, PiecewiseConstant, floor_time_to_minute, floor_time_to_second, replace_orientation_matrices, string_to_timedelta
from visualization import SatelliteVisualizer


class Simulation:
    def __init__(self, sat: Spacecraft, controller: Controller, tle: tuple[str, str], initial_attitude_BO: R, initial_ang_vel_B: np.ndarray,
                 dt: datetime.timedelta, t0: datetime.datetime, tf: datetime.datetime,
                 enable_viz: bool = True, enable_log: bool = True, enable_disturbance_torques: bool = True,
                 enable_disturbance_forces: bool = True):
        
        self.controller = controller

        self.t0 = t0
        self.dt = dt
        self.tf = tf

        self.sat = sat
        self.inital_state = np.zeros(22)

        orbit_model = SGP4.from_tle(*tle)
        self.initial_tle = tle
        initial_r_ECI, initial_v_ECI = orbit_model.propagate(t0)
        self.inital_state[:3] = initial_r_ECI
        self.inital_state[3:6] = initial_v_ECI

        # R_BI = R_BO * (R_IO)^-1
        self.inital_state[6:10] = (initial_attitude_BO * orc_to_eci(self.inital_state[0:3],
                                                                    self.inital_state[3:6]).inv()
                                   ).as_quat(scalar_first=False)
        

        self.enable_disturbance_torques = enable_disturbance_torques
        self.enable_disturbance_forces = enable_disturbance_forces
        # use these two together enable actuators and freeze actuator states
        self.enable_actuators = False
        self.freeze_actuator_states = False
        self.freeze_body_rates = False

        if initial_ang_vel_B is not None:
            self.inital_state[10:13] = initial_ang_vel_B

        self.enable_log = enable_log
        if self.enable_log:
            self.log_folder = "Simulation_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)
                
            with open(os.path.join(self.log_folder, ".gitignore"), "w") as f:
                f.write("*")
                
            self.to_json(os.path.join(self.log_folder, "config.json"))

            self.state_logger = Logger(os.path.join(self.log_folder, "state.csv"),
                                       ['t', 'r_eci_x', 'r_eci_y', 'r_eci_z', 'v_eci_x', 'v_eci_y', 'v_eci_z',
                                        'q_BI_x', 'q_BI_y', 'q_BI_z', 'q_BI_w', 'omega_x', 'omega_y', 'omega_z',
                                        'omega_rw_1', 'omega_rw_2', 'omega_rw_3', 'i_mag_1', 'i_mag_2', 'i_mag_3',
                                        'i_rw_1', 'i_rw_2', 'i_rw_3'])
            self.input_logger = Logger(os.path.join(self.log_folder, "input.csv"),
                                       ['t', 'u_mag_1', 'u_mag_2', 'u_mag_3', 'u_rw_1', 'u_rw_2', 'u_rw_3',
                                        'i_cmd_mag_1', 'i_cmd_mag_2', 'i_cmd_mag_3', 'i_cmd_rw_1', 'i_cmd_rw_2', 'i_cmd_rw_3'])
            self.env_logger = Logger(os.path.join(self.log_folder, "environment.csv"), 
                                     ['t', 'rho', 'B_x', 'B_y', 'B_z', 'sun_pos_x', 'sun_pos_y', 'sun_pos_z', 
                                      'in_shadow', 'moon_pos_x', 'moon_pos_y', 'moon_pos_z'])
            self.mea_logger = Logger(os.path.join(self.log_folder, "measurements.csv"),
                                       ['t', 'sun_x', 'sun_y', 'sun_z', 'mag_x', 'mag_y', 'mag_z', 'gps_x', 'gps_y', 'gps_z',
                                        'gyro_x', 'gyro_y', 'gyro_z', 'omega_rw_1', 'omega_rw_2', 'omega_rw_3'])
            self.est_logger = Logger(os.path.join(self.log_folder, "estimation.csv"),
                                       ['t', 'r_eci_x', 'r_eci_y', 'r_eci_z', 'v_eci_x', 'v_eci_y', 'v_eci_z',
                                        'q_BI_x', 'q_BI_y', 'q_BI_z', 'q_BI_w', 'omega_x', 'omega_y', 'omega_z',
                                        'h_rw_x', 'h_rw_y', 'h_rw_z',
                                        'omega_rw_1', 'omega_rw_2', 'omega_rw_3',
                                        'i_mag_1', 'i_mag_2', 'i_mag_3', 'i_rw_1', 'i_rw_2', 'i_rw_3'])


        self.att_ekf = AttitudeEKF(q0=self.inital_state[6:10], b0=self.inital_state[10:13], P0=np.eye(6), Qc=np.eye(6),
                                   R_sun=np.eye(3), R_mag=np.eye(3))

        # Sun and moon position change very slowly. For performance a new value is only calculated every minute
        self.sun_position = PiecewiseConstant(fn=env.sun_position, time_bucket_fn=floor_time_to_minute)
        self.moon_position = PiecewiseConstant(fn=env.moon_position, time_bucket_fn=floor_time_to_minute)

        # Heavy(ish) environment calls, cache them at 1 Hz for speed
        self.atmosphere_density = PiecewiseConstant(fn=env.atmosphere_density_msis, time_bucket_fn=floor_time_to_second)
        self.magnetic_field = PiecewiseConstant(fn=env.magnetic_field_vector, time_bucket_fn=floor_time_to_second)

        self.enable_viz = enable_viz
        if self.enable_viz:
            self.viz = SatelliteVisualizer(sat.surfaces)

    @classmethod
    def from_json(cls, file_path: str, enable_viz: bool = True, enable_log: bool = True):
        with open(file_path, "r") as f:
            data = json.load(f)

        controller = eval("controllers." + data["Controller"]["name"])(**data["Controller"]["params"])

        data_sim = data["Simulation"]
        t0 = datetime.datetime.fromisoformat(data_sim["Start"])
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=datetime.timezone.utc)

        dt = string_to_timedelta(data_sim["Stepsize"])

        dur = string_to_timedelta(data_sim["Duration"])

        tf = t0 + dur

        enable_disturbance_torques = data_sim.get("DisturbanceTorques", True)
        enable_disturbance_forces = data_sim.get("DisturbanceForces", True)

        data_init_state = data["InitialState"]

        tle1 = data_init_state["TLE"]["Line 1"]
        tle2 = data_init_state["TLE"]["Line 2"]
        orbit_model = SGP4.from_tle(tle1, tle2)
        r_ECI, v_ECI = orbit_model.propagate(t0)

        roll = data_init_state["Attitude"]["Roll (deg)"]
        pitch = data_init_state["Attitude"]["Pitch (deg)"]
        yaw = data_init_state["Attitude"]["Yaw (deg)"]

        R_BO = euler_ocr_to_sbc(roll, pitch, yaw)

        ang_vel_B_BO = np.deg2rad(data_init_state["AngularVelocity (wrt ORC in SBC) (deg/s)"])

        orbit_ang_vel = np.linalg.norm(v_ECI)/np.linalg.norm(r_ECI)
        init_ang_vel_B_BI = ang_vel_B_BO + R_BO.apply(np.array((0, -orbit_ang_vel, 0)))

        return cls(Spacecraft.from_dict(data["SpacecraftParams"]), controller, (tle1, tle2), R_BO,
                   init_ang_vel_B_BI, dt, t0, tf, enable_viz, enable_log, enable_disturbance_torques, enable_disturbance_forces)

    def to_json(self, file_path: str):
        data = {}

        data["Simulation"] = {
            "Start": self.t0.isoformat(),
            "Stepsize": str(self.dt),
            "Duration": str(self.tf - self.t0),
            "DisturbanceTorques": self.enable_disturbance_torques,
            "DisturbanceForces": self.enable_disturbance_forces
        }

        data["Controller"] = {
            "name": self.controller.__class__.__name__, 
            "params": self.controller.to_dict()
            }
        

        r_eci = self.inital_state[0:3]
        v_eci = self.inital_state[3:6]
        q_BI = self.inital_state[6:10]
        omega_BI_B = self.inital_state[10:13]

        R_BO = orc_to_sbc(q_BI, r_eci, v_eci)

        pitch, roll, yaw = R_BO.as_euler('XYZ', degrees=True)

        orbit_ang_vel = np.linalg.norm(v_eci) / np.linalg.norm(r_eci)
        omega_OI_B = R_BO.apply(np.array([0, -orbit_ang_vel, 0]))
        ang_vel_B_BO = omega_BI_B - omega_OI_B

        data["InitialState"] = {
            "TLE": {
                "Line 1": self.initial_tle[0],
                "Line 2": self.initial_tle[1]
            },
            "Attitude": {
                "Roll (deg)": float(roll),
                "Pitch (deg)": float(pitch),
                "Yaw (deg)": float(yaw)
            },
            "AngularVelocity (wrt ORC in SBC) (deg/s)": np.rad2deg(ang_vel_B_BO).tolist()
        }

        data["SpacecraftParams"] = self.sat.to_dict()

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def run(self):
        state = self.inital_state
        t = self.t0

        with tqdm(total=(self.tf - self.t0).total_seconds()/60, desc="Simulation time", unit="sim min") as pbar:
            while t <= self.tf:

                # TODO: find cleaner solution for updating sensors and visualization
                r_eci = state[0:3]
                v_eci = state[3:6]
                q_BI = state[6:10]
                omega = state[10:13]
                omega_rws = state[13:16]
                mag_curr = state[16:19]
                rws_curr = state[19:22]

                R_BO = orc_to_sbc(q_BI, r_eci, v_eci)
                R_OI = orc_to_eci(r_eci, v_eci).inv()
                R_BI = eci_to_sbc(q_BI)

                lat, lon, alt = eci_to_geodedic(r_eci)

                rho: float = self.atmosphere_density(t, lat, lon, alt)  # type: ignore
                B: np.ndarray = self.magnetic_field(t, lat, lon, alt)  # type: ignore

                sun_pos: np.ndarray = self.sun_position(t)  # type: ignore
                in_shadow = env.is_in_shadow(r_eci, sun_pos)
                moon_pos: np.ndarray = self.moon_position(t)  # type: ignore

                self.sat.sun_sensor.measure(t, sun_pos)
                self.sat.magnetometer.measure(t, B)
                self.sat.gps.measure(t, r_eci)
                self.sat.gyro.measure(t, omega)
                for i, rw in enumerate(self.sat.rw_speed_sensors):
                    rw.measure(t, omega_rws[i])

                # Flight software (FSW)
                sun_mea, new_sun_mea = self.sat.sun_sensor.read(t)
                mag_mea, new_mag_mea = self.sat.magnetometer.read(t)
                gps_mea, new_gps_mea = self.sat.gps.read(t)
                gyro_mea, new_gyro_mea = self.sat.gyro.read(t)
                omega_rw_mea, new_omega_rw_mea = zip(*[rw.read(t) for rw in self.sat.rw_speed_sensors])

                # self.orbital_estimator.pedict(t)
                self.att_ekf.predict(t, gyro_mea)

                # if new_gps_mea:
                #     self.orbital_estimator.update_gps(t, gps_mea)
                r_eci_est, v_eci_est = state[:3], state[3:6]  # self.orbital_estimator.get_state()

                if new_sun_mea:
                    self.att_ekf.update_sun(t, sun_mea)
                if new_mag_mea:
                    self.att_ekf.update_mag(t, mag_mea, r_eci_est)

                omega_est = omega # TODO: get estimators working!!
                omega_rw_est = omega_rw_mea
                curr_rw_est = rws_curr
                mag_curr_est = mag_curr

                h_w = np.zeros(3)
                for i, rw in enumerate(self.sat.rws):
                    omega_parallel_body = float(np.dot(rw.axis, omega_est)) 
                    h_w += rw.inertia * (omega_rw_est[i] + omega_parallel_body) * rw.axis

                state_est = np.concatenate((q_BI, omega, h_w)) # 

                u = self.controller.calc_input_cmds(state_est, np.concatenate((r_eci_est, v_eci_est)))
                current_cmds = to_current_commands(u, B, self.sat.mag, self.sat.rws)

                if self.enable_log:
                    self.state_logger.log([t] + list(state))
                    self.input_logger.log([t] + list(u) + list(current_cmds))
                    self.env_logger.log([t, rho] + list(B) + list(sun_pos) + [in_shadow] + list(moon_pos))
                    self.mea_logger.log([t] + list(sun_mea) + list(mag_mea) + list(gps_mea) + list(gyro_mea) + list(omega_rw_mea))
                    self.est_logger.log([t] + list(r_eci_est) + list(v_eci_est) + list(state_est) + 
                                        list(omega_rw_est) + list(mag_curr_est) + list(curr_rw_est))


                next_state = rk4_step(self.world_dynamics, state, current_cmds, t, self.dt)

                next_state[6:10] /= np.linalg.norm(next_state[6:10])  # normalize quaternion

                t += self.dt
                state = next_state

                if self.enable_viz and int((t - self.t0).total_seconds()) % 3 == 0:
                    self.viz.update(self.sat.surfaces, R_BO, R_OI, R_OI.apply(v_eci))
                pbar.update(self.dt.total_seconds() / 60)

    def world_dynamics(self, x: np.ndarray, u: np.ndarray, t: datetime.datetime):
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
        mag_curr = x[16:19]
        rws_curr = x[19:22]

        u_mag = u[:3]
        u_rw = u[3:6]

        R_BO = orc_to_sbc(q_BI, r_eci, v_eci)
        R_BI = eci_to_sbc(q_BI)

        lat, lon, alt = eci_to_geodedic(r_eci)

        rho: float = self.atmosphere_density(t, lat, lon, alt)  # type: ignore
        B: np.ndarray = self.magnetic_field(t, lat, lon, alt)  # type: ignore

        sun_pos: np.ndarray = self.sun_position(t)  # type: ignore
        in_shadow = env.is_in_shadow(r_eci, sun_pos)
        moon_pos: np.ndarray = self.moon_position(t)  # type: ignore

        F_grav = dis.non_spherical_gravity_forces(r_eci, self.sat.m)
        F_third = dis.third_body_forces(r_eci, self.sat.m, sun_pos, moon_pos)
        tau_gg = dis.gravity_gradient(r_eci, R_BO, self.sat.J_B)
        F_aero, tau_aero = dis.aerodynamic_drag(r_eci, v_eci, R_BI, self.sat.surfaces, rho)
        F_SRP, tau_SRP = dis.solar_radiation_pressure(r_eci, sun_pos, in_shadow, R_BI, self.sat.surfaces)

        if not self.enable_disturbance_torques:
            tau_gg = np.zeros(3)
            tau_aero = np.zeros(3)
            tau_SRP = np.zeros(3)

        if not self.enable_disturbance_forces:
            F_grav = np.zeros(3)
            F_third = np.zeros(3)
            F_aero = np.zeros(3)
            F_SRP = np.zeros(3)

        if self.enable_actuators:
            tau_rw, h_rw = sum([np.array(rw.torque_ang_momentum(rws_curr[i], omega_rws[i], omega))
                               for i, rw in enumerate(self.sat.rws)])  # type: ignore
            tau_mag = sum([np.array(mag.torque(mag_curr[i], B)) for i, mag in enumerate(self.sat.mag)])
        else:
            tau_rw = np.zeros(3)
            h_rw = np.zeros(3)
            tau_mag = np.zeros(3)

        # differential equations
        d_r = v_eci
        d_v = self.sat.orbit_dynamics(r_eci, R_BI.inv().apply(F_aero + F_SRP) + F_third + F_grav)
        d_q = quaternion_kinematics(q_BI, omega)
        d_omega = self.sat.attitude_dynamics(omega, h_rw, tau_mag - tau_rw, tau_gg + tau_aero + tau_SRP)
        if self.freeze_body_rates is True:
            d_omega = np.zeros(3)

        d_curr_mag = np.array([mag.dynamics(u_mag[i], mag_curr[i]) for i, mag in enumerate(self.sat.mag)])
        d_omega_rw, d_curr_rw = np.array([rw.dynamics(u_rw[i], d_omega, rws_curr[i]) for i, rw in enumerate(self.sat.rws)]).T

        if self.freeze_actuator_states is True:
            d_omega_rw = np.zeros(3)
            d_curr_mag = np.zeros(3)
            d_curr_rw = np.zeros(3)

        dx = np.concatenate((d_r, d_v, d_q, d_omega, d_omega_rw, d_curr_mag, d_curr_rw))

        return dx


def rk4_step(f: Callable[[np.ndarray, np.ndarray, datetime.datetime], np.ndarray],
             x: np.ndarray, u: np.ndarray, t: datetime.datetime, dt: datetime.timedelta) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        State after one time step.
    """

    dt_float = dt.total_seconds()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=Warning)
            k1 = f(x, u, t)
            k2 = f(x + 0.5 * dt_float * k1, u, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt_float * k2, u, t + 0.5 * dt)
            k4 = f(x + dt_float * k3, u, t + dt)

            x_next = x + (dt_float / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    except Warning as e:
        print(f"RuntimeWarning caught in rk4_step: {e}")
        print(f"t: {t}, dt: {dt_float}")
        print(f"x: {x}")
        print(f"u: {u}")
        if 'k1' in locals(): print(f"k1: {k1}")
        if 'k2' in locals(): print(f"k2: {k2}")
        if 'k3' in locals(): print(f"k3: {k3}")
        if 'k4' in locals(): print(f"k4: {k4}")
        raise e
    
    return x_next


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "simulation_config.json")

    sim = Simulation.from_json(file_path, enable_viz=False, enable_log=True)
    sim.run()
