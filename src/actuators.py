import numpy as np


class ReactionWheel:
    def __init__(
        self,
        max_torque: float,
        max_rpm: float,
        inertia: float,
        axis: np.ndarray,
        max_current: float = 1.0,
        tau_current: float = 0.1,
        torque_constant: float | None = None,
    ) -> None:
        # to float (scalars)
        self.max_torque = float(max_torque)
        self.max_rpm = float(max_rpm)
        self.inertia = float(inertia)

        # normalise spin axis to unit vector in body frame
        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            raise ValueError("ReactionWheel axis must be non-zero!")
        self.axis = axis / norm

        # simple current model (first-order)
        self.max_current = float(max_current)
        self.tau_current = float(tau_current)

        # torque constant K_t
        if torque_constant is None:
            # choose K_t so that max_current -> max_torque
            self.K_t = self.max_torque / self.max_current
        else:
            self.K_t = float(torque_constant)

        # store mech. speed in rad/s
        self.max_omega = 2.0 * np.pi * self.max_rpm / 60.0

    def to_dict(self):
        data = self.__dict__
        data["axis"] = self.axis.tolist()
        return data

    # algebraic way of calculating the moment vector affecting the RW
    def torque_ang_momentum(
        self,
        i: float,
        omega_w: float,
        omega_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        # making sure it's 3-sized 1D vector
        omega_body = np.asarray(omega_body, dtype=float).reshape(3)
        # saturate current
        i_sat = float(np.clip(i, -self.max_current, self.max_current))
        # current into torque then saturation
        tau_m = self.K_t * i_sat
        tau_m = float(np.clip(tau_m, -self.max_torque, self.max_torque))
        # torque vector along spin axis, applied to the wheel
        tau_rw = tau_m * self.axis
        # internal angular momentum of this wheel in body frame:
        omega_parallel_body = float(np.dot(self.axis, omega_body))
        h_wheel = self.inertia * (float(omega_w) + omega_parallel_body) * self.axis

        return tau_rw, h_wheel

    # differential equations for wheel speed and current
    def dynamics(
        self,
        u: float,
        omega_dot_body: np.ndarray,
        i: float,
    ) -> tuple[float, float]:

        # making sure it's 3-sized 1D vector
        omega_dot_body = np.asarray(omega_dot_body, dtype=float).reshape(3)
        # commanded current (saturated)
        i_cmd = float(np.clip(u, -self.max_current, self.max_current))
        # simple first-order electrical response:
        di_dt = (i_cmd - float(i)) / self.tau_current
        # using (saturated) current for torque generation
        i_eff = float(np.clip(i, -self.max_current, self.max_current))
        tau_m = self.K_t * i_eff
        tau_m = float(np.clip(tau_m, -self.max_torque, self.max_torque))

        # mechanical dynamics (no friction)
        # => dot{Omega} = tau_m / J_w - s^T dot{omega}_B
        domega_w = tau_m / self.inertia - float(np.dot(self.axis, omega_dot_body))

        return domega_w, di_dt


class Magnetorquer:
    def __init__(
        self,
        max_moment: float,
        axis: np.ndarray,
        max_current: float = 1.0,
        tau_current: float = 0.01,
    ) -> None:
        self.max_moment = float(max_moment)
        # making sure it's 3-sized 1D vector
        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            raise ValueError("Magnetorquer axis must be non-zero!")
        self.axis = axis / norm

        self.max_current = float(max_current)
        self.tau_current = float(tau_current)

    def to_dict(self):
        data = self.__dict__
        data["axis"] = self.axis.tolist()
        return data


    # magnetic torque calculations
    def torque(self, i: float, B_body: np.ndarray) -> np.ndarray:
        # making sure it's 3-sized 1D vector
        B_body = np.asarray(B_body, dtype=float).reshape(3)
        # saturate current and corresponding dipole magnitude
        i_sat = float(np.clip(i, -self.max_current, self.max_current))
        m_scalar = self.max_moment * (i_sat / self.max_current)
        # dipole along axis
        m_vec = m_scalar * self.axis
        # torque tau = m × B
        tau_mag = np.cross(m_vec, B_body)
        return tau_mag

    # differential equations for magnetorquers
    def dynamics(self, u: float, i: float) -> float:
        i_cmd = float(np.clip(u, -self.max_current, self.max_current))
        di_dt = (i_cmd - float(i)) / self.tau_current
        return di_dt