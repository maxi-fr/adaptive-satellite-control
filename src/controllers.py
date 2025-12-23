from abc import ABC, abstractmethod
import casadi as ca
import numpy as np
from typing import Callable, List, Tuple, Optional, Union, Any
import control as ct


class Controller(ABC):
    """
    Abstract base class for controllers.
    """
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def calc_input_cmds(self, w: np.ndarray, x: np.ndarray, dt: Optional[float] = None, *args: Any) -> np.ndarray:
        """
        Calculates the control input commands.

        Parameters
        ----------
        w : np.ndarray
            Reference vector.
        x : np.ndarray
            State vector.
        dt : float, optional
            Time step, by default None.

        Returns
        -------
        np.ndarray
            Control input vector.
        """
        pass


class PI(Controller):
    """
    Proportional-Integral controller with anti-windup.
    """
    
    def __init__(self, K_q: np.ndarray, K_omega: np.ndarray, K_w: np.ndarray, K_q_int: np.ndarray, 
                 operating_point: Tuple[np.ndarray, np.ndarray], m: float, u_min: float, u_max: float) -> None:
        """
        Initializes the PI controller.

        Parameters
        ----------
        K_q : np.ndarray
            Proportional gain matrix for attitude error.
        K_omega : np.ndarray
            Derivative gain matrix for angular velocity error.
        K_w : np.ndarray
            Gain matrix for wheel momentum error.
        K_q_int : np.ndarray
            Integral gain matrix for attitude error.
        operating_point : Tuple[np.ndarray, np.ndarray]
            The operating point (x_star, u_star).
        m : float
            Anti-windup gain.
        u_min : float
            Minimum control input value (saturation).
        u_max : float
            Maximum control input value (saturation).
        """


        super().__init__()
        self.x_star, self.u_star = *operating_point
        self.q_err_int = np.zeros(3)

        self.K_q = K_q
        self.K_omega = K_omega
        self.K_wheel = K_w
        self.K_q_int = K_q_int
        self.u_min = u_min
        self.u_max = u_max
        self.m = m

    @classmethod
    def from_lqr(cls, f_x: Callable[[np.ndarray, np.ndarray], np.ndarray], f_u: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                 operating_point: Tuple[np.ndarray, np.ndarray], Q: np.ndarray, R: np.ndarray, 
                 m: float, u_min: float, u_max: float, K_q_int: np.ndarray|None = None) -> "PI":
        """
        Creates a PI controller using LQR gains.

        Parameters
        ----------
        f_x : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Function returning the Jacobian of dynamics w.r.t state (A matrix).
        f_u : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Function returning the Jacobian of dynamics w.r.t input (B matrix).
        operating_point : Tuple[np.ndarray, np.ndarray]
            The operating point (x_star, u_star).
        Q : np.ndarray
            State cost matrix.
        R : np.ndarray
            Input cost matrix.
        m : float
            Anti-windup gain.
        u_min : float
            Minimum control input.
        u_max : float
            Maximum control input.
        K_q_int : np.ndarray, optional
            Integral gain matrix for attitude error, by default None.

        Returns
        -------
        PI
            Initialized PI controller instance.
        """
        A = f_x(*operating_point)
        B = f_u(*operating_point)
        
        K, S, E = ct.lqr(A, B, Q, R)

        if K_q_int is None:
            K_q_int = np.zeros((K.shape[0], 3))

        return cls(K[:, :3], K[:, 3:6], K[:, 6:9], K_q_int, operating_point, m, u_min, u_max)
    

    def calc_input_cmds(self, w: np.ndarray, x: np.ndarray, dt: Optional[float] = None, *args: Any) -> np.ndarray:
        """
        Calculates the control input.

        Parameters
        ----------
        w : np.ndarray
            Reference vector (unused in this implementation as x_star is fixed).
        x : np.ndarray
            Current state vector.
        dt : float, optional
            Time step, required for integral term.

        Returns
        -------
        np.ndarray
            Control input vector.

        Raises
        ------
        ValueError
            If dt is not provided.
        """

        if dt is None:
            raise ValueError("dt must be provided")
        D_x = x - self.x_star
        q_err = D_x[:3]
        omega_err = D_x[3:]
        h_w = D_x[6:]


        self.q_err_int += q_err * dt

        u_R = - (self.K_q @ q_err + self.K_omega @ omega_err + self.K_wheel @ h_w + self.K_q_int @ self.q_err_int)

        u = np.clip(u_R, self.u_min, self.u_max)

        self.q_err_int -= self.m * (u_R - u) # anti wind up

        return u + self.u_star
    

class GainScheduling(Controller):

    def __init__(self, f: ca.Function, f_jac_x: ca.Function, f_jac_u: ca.Function, 
                 x_rho: Callable[[np.ndarray], np.ndarray], w_rho: Callable[[np.ndarray], np.ndarray], 
                 u_rho: Callable[[np.ndarray], np.ndarray], 
                 rho: List[np.ndarray], calc_scheduling_param: Callable[[np.ndarray], np.ndarray],
                 Q: List[np.ndarray]|np.ndarray, R: List[np.ndarray]|np.ndarray):
        """
        Initializes the Gain Scheduling controller.

        Parameters
        ----------
        f : ca.Function
            System dynamics function.
        f_jac_x : ca.Function
            Jacobian of dynamics w.r.t state.
        f_jac_u : ca.Function
            Jacobian of dynamics w.r.t input.
        x_rho : Callable[[np.ndarray], np.ndarray]
            Function mapping scheduling parameter to state operating point.
        w_rho : Callable[[np.ndarray], np.ndarray]
            Function mapping scheduling parameter to reference operating point.
        u_rho : Callable[[np.ndarray], np.ndarray]
            Function mapping scheduling parameter to input operating point.
        rho : List[np.ndarray]
            List of scheduling parameter values defining the operating points.
        calc_scheduling_param : Callable[[np.ndarray, *args], np.ndarray]
            Function to calculate the current scheduling parameter from state.
        Q : Union[List[np.ndarray], np.ndarray]
            State cost matrix or list of matrices for each operating point used in LQR.
        R : Union[List[np.ndarray], np.ndarray]
            Input cost matrix or list of matrices for each operating point used in LQR.
        """

        super().__init__()
        self.operating_points = [(x_rho(p), u_rho(p), w_rho(p)) for p in rho]
        self.rho = np.stack([np.atleast_1d(p) for p in rho], axis=0)
        self.calc_scheduling_param = calc_scheduling_param

        self.f = f

        self.Q = Q if isinstance(Q, list) else [Q] * len(self.operating_points)
        self.R = R if isinstance(R, list) else [R] * len(self.operating_points)

        # for x, u, w in self.operating_points:
        #     print(f"Operating Point: x={x}, u={u}, w={w}")

        self.linear_models = [(f_jac_x(x, u), f_jac_u(x, u)) for x, u, w in self.operating_points]

        #self.place_gains = [ct.place(np.array(A).squeeze(), np.array(B).squeeze(), P[i]) for i, (A, B) in enumerate(self.linear_models)]
        self.lqr_gains = [ct.lqr(np.squeeze(A), np.squeeze(B), self.Q[i], self.R[i])[0] for i, (A, B) in enumerate(self.linear_models)] # type: ignore

    def closest_operating_points(self, beta: np.ndarray) -> Tuple[int, int]:
        """
        Finds the indices of the two closest operating points.

        Parameters
        ----------
        beta : np.ndarray
            Current scheduling parameter.

        Returns
        -------
        Tuple[int, int]
            Indices of the two closest operating points.
        """
        i, j = np.argsort(np.linalg.norm(self.rho - beta, axis=1))[:2]

        return i, j


    def calc_input_cmds(self, w: np.ndarray, x: np.ndarray, dt: Optional[float] = None, *args: Any) -> np.ndarray:
        """
        Calculates the control input using gain scheduling.

        Parameters
        ----------
        w : np.ndarray
            Reference vector.
        x : np.ndarray
            Current state vector.
        dt : float, optional
            Time step (unused).

        Returns
        -------
        np.ndarray
            Control input vector.
        """
        beta = self.calc_scheduling_param(x, *args)

        i, j = self.closest_operating_points(beta)

        delta_u_i = np.dot(self.lqr_gains[i], ((w - self.operating_points[i][2]) - (x - self.operating_points[i][0])))
        delta_u_j = np.dot(self.lqr_gains[j], ((w - self.operating_points[j][2]) - (x - self.operating_points[j][0])))

        alpha = np.dot(self.rho[i] - self.rho[j], beta - self.rho[j]) / np.linalg.norm(self.rho[i] - self.rho[j])

        delta_u = delta_u_i * alpha + (1-alpha) * delta_u_j

        u_i = self.operating_points[i][1]
        
        u = delta_u + u_i

        return u

if __name__ == "__main__":

    x = ca.SX.sym("x", 2)
    dx = ca.SX.sym("dx", 2)
    u = ca.SX.sym("u")
    p = ca.SX.sym("p")

    dx = - x**3 + np.ones(2)*u
    f = ca.Function("f", [x, u], [dx], ["x", "u"], ["dx"])

    f_jac_x = ca.Function("f_jac_x", [x, u], [ca.jacobian(dx, x)], ["x", "u"], ["jac_x"])
    f_jac_u = ca.Function("f_jac_u", [x, u], [ca.jacobian(dx, u)], ["x", "u"], ["jac_u"])


    rho = np.arange(-2, 3)

    u_rho = lambda p: p # ca.Function("u_rho", [p], [p], ["rho"], ["u"])
    x_rho = lambda p: np.array((np.sign(p)*np.abs(p)**(1/3), np.sign(p)*np.abs(p)**(1/3))) # ca.Function("x_rho", [p], [ca.sign(p)*ca.fabs(p)**(1/3), ca.sign(p)*ca.fabs(p)**(1/3)], ["rho"], ["x1", "x2"])
    w_rho = lambda p : ca.sign(p)*ca.fabs(p)**(1/3) # ca.Function("w_rho", [p], [ca.sign(p)*ca.fabs(p)**(1/3)], ["rho"], ["w"])

    c = lambda x: x[0]**3
    
    gs = GainScheduling(f, f_jac_x, f_jac_u, x_rho, w_rho, u_rho, rho, c, Q, R)

    # print(gs.linear_models)
    # print(gs.lqr_gains)

    t = np.arange(0, 2.5, 0.01)
    w = 1.3 * np.ones_like(t)
    x = np.zeros((t.size, 2))
    u = np.zeros_like(t)
    for i in range(t.size-1):
        u[i] = gs.calc_input_cmds(w[i], x[i])
        
        x[i+1] = x[i] + 0.01 * np.array(f(x[i], u[i])).item(0)

    u[-1] = u[-2]
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, w, linestyle="--")
    ax1.plot(t, x)
    ax2.plot(t, u)
    ax1.grid()
    ax2.grid()
    plt.show()