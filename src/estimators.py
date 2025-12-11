import numpy as np
import datetime
from scipy.spatial.transform import Rotation as R
import environment as env
from kinematics import quaternion_kinematics, eci_to_geodedic

def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y, x,  0.0]])

class AttitudeEKF:
    # Filtering the orientation of the satellite + base drift of gyro
    def __init__(
        self,
        q0: np.ndarray,             # initial guesses for quaternions
        b0: np.ndarray,             # initial guesses for bias of gyro
        P0: np.ndarray,             # initial covariance error 6x6
        Qc: np.ndarray,             # spectral density of noise 6x6
        R_sun: np.ndarray,
        R_mag: np.ndarray,          # covariances of the Sun sensor and magnetometer
    ) -> None:
        # making sure it's a 4-element vector
        q0 = np.asarray(q0, dtype=float).reshape(4)
        # quaternion normalization (to represent rotation)
        self.q = q0 / np.linalg.norm(q0)
        self.b = np.asarray(b0, dtype=float).reshape(3) #gyro bias
        # covariance and noise matrices
        self.P = np.asarray(P0, dtype=float).reshape(6, 6)
        self.Qc = np.asarray(Qc, dtype=float).reshape(6, 6)      # process noise spectral density
        self.R_sun = np.asarray(R_sun, dtype=float).reshape(3, 3)
        self.R_mag = np.asarray(R_mag, dtype=float).reshape(3, 3)

        # saving the time of last prediction
        self.t_last: datetime.datetime | None = None

    # prediction step
    def predict(self, t: datetime.datetime, omega_meas: np.ndarray) -> None:
        # Time update using gyro measurement omega_meas (rad/s)
        if self.t_last is None:
            # initialise time reference but not propagating
            self.t_last = t
            return
        # calculating the discrete time step
        dt = (t - self.t_last).total_seconds()
        if dt <= 0.0:
            # non-positive time step: ignore to avoid singular behaviour
            self.t_last = t
            return
        self.t_last = t
        #making the angle velocity a 3-element vector
        omega_meas = np.asarray(omega_meas, dtype=float).reshape(3)
        # bias-compensated angular rate
        omega_eff = omega_meas - self.b
        # propagate quaternion (nominal attitude)
        dqdt = quaternion_kinematics(self.q, omega_eff)
        self.q = self.q + dqdt * dt
        self.q /= np.linalg.norm(self.q)

        # getting the continuous F and G matrices for our error model (omega_eff)
        F, G = self._continuous_FG(omega_eff)
        Phi = np.eye(6) + F * dt      # state transition approx
        Qd = G @ self.Qc @ G.T * dt   # discrete process noise (Euler approx)
        # EKF porpagation
        self.P = Phi @ self.P @ Phi.T + Qd
        # enforce symmetry numerically
        self.P = 0.5 * (self.P + self.P.T)

    # Construct continuous-time F and G for the 6-state error model
    def _continuous_FG(self, omega_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega_eff = np.asarray(omega_eff, dtype=float).reshape(3)
        # input being effective angle velocity
        F11 = -skew(omega_eff)
        F12 = -np.eye(3)
        F = np.block([
            [F11, F12],
            [np.zeros((3, 3)), np.zeros((3, 3))],
        ])

        G = np.block([
            [-np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.eye(3)],
        ])

        return F, G

    # GENERIC MEASUREMENT UPDATE
    def _update(
        self,
        z_meas: np.ndarray,
        z_pred: np.ndarray,
        H: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        # Generic EKF measurement update for a 3D vector measurement
        z_meas = np.asarray(z_meas, dtype=float).reshape(3)
        z_pred = np.asarray(z_pred, dtype=float).reshape(3)
        H = np.asarray(H, dtype=float).reshape(3, 6)
        R_meas = np.asarray(R_meas, dtype=float).reshape(3, 3)
        # innovation
        y = z_meas - z_pred
        # innovation covariance
        S = H @ self.P @ H.T + R_meas
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # error-state update
        dx = K @ y
        dtheta = dx[:3]
        db = dx[3:]
        # update bias
        self.b = self.b + db
        # update quaternion using small-angle rotation
        dq_vec = 0.5 * dtheta
        dq = np.hstack((dq_vec, [1.0]))
        dq /= np.linalg.norm(dq)
        # converting existing quaternion and correction into rotation
        q_rot = R.from_quat(self.q)
        dq_rot = R.from_quat(dq)
        self.q = (dq_rot * q_rot).as_quat()
        # Joseph-form covariance update for better numerical stability
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # SUN SENSOR UPDATE
    def update_sun(self, t: datetime.datetime, sun_body_meas: np.ndarray) -> None:
        # making sure it's a 3-element vector
        sun_body_meas = np.asarray(sun_body_meas, dtype=float).reshape(3)
        # inertial Sun direction (ECI/GCRS) from environment model
        r_sun_I = env.sun_position(t)
        s_I = r_sun_I / np.linalg.norm(r_sun_I)
        # from quaternion, we make the rotational matrix
        C_BI = R.from_quat(self.q).as_matrix()
        # predicted direction of Sun
        s_B_pred = C_BI @ s_I
        # normalise both measured and predicted vectors
        s_B_pred /= np.linalg.norm(s_B_pred)
        sun_body_meas /= np.linalg.norm(sun_body_meas)
        # measurement matrix
        H = np.hstack((skew(s_B_pred), np.zeros((3, 3))))
        # updating sun measurements
        self._update(sun_body_meas, s_B_pred, H, self.R_sun)

    # update from the magnometer
    def update_mag(
        self,
        t: datetime.datetime,
        B_body_meas: np.ndarray,
        r_ECI: np.ndarray,
    ) -> None:
        B_body_meas = np.asarray(B_body_meas, dtype=float).reshape(3)
        r_ECI = np.asarray(r_ECI, dtype=float).reshape(3)
        # convert ECI position to geodetic coordinates for the IGRF model
        lat, lon, alt = eci_to_geodedic(r_ECI)
        # magnetic field in ECI from the environment model
        B_I = env.magnetic_field_vector(t, lat, lon, alt)
        # predicted magnetic field in body frame
        C_BI = R.from_quat(self.q).as_matrix()
        B_B_pred = C_BI @ B_I
        # measurement matrix
        H = np.hstack((skew(B_B_pred), np.zeros((3, 3))))
        self._update(B_body_meas, B_B_pred, H, self.R_mag)

    # accessor (returns the copy of the quaternion, bias and covariance)
    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.q.copy(), self.b.copy(), self.P.copy()