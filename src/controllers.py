from abc import ABC, abstractmethod
import casadi as ca
import numpy as np
from typing import Callable, List, Tuple, Optional, Union, Any
import control as ct
from scipy.spatial.transform import Rotation as R

from kinematics import orc_to_sbc
from utils import cgi_allocation


class Controller(ABC):
    """
    Abstract base class for controllers.
    """
    def __init__(self) -> None:
        self.u_min = -np.inf
        self.u_max = np.inf

    def update_actuator_limits(self, u_min: np.ndarray, u_max: np.ndarray) -> None:
        """
        Updates the actuator limits.

        Parameters
        ----------
        u_min : np.ndarray
            Minimum control input vector.
        u_max : np.ndarray
            Maximum control input vector.
        """
        self.u_min = u_min
        self.u_max = u_max  

    def calc_nadir_state_error(self, state_est: np.ndarray, orbit_state: np.ndarray) -> np.ndarray:
        r_eci, v_eci = orbit_state[:3], orbit_state[3:6]

        R_BO = orc_to_sbc(state_est[:4], r_eci, v_eci) 

        q_err_full = R_BO.as_quat(scalar_first=False) # TODO: do we have to check if q_w is positive?

        omega_0 = np.linalg.norm(v_eci)/ np.linalg.norm(r_eci)
        omega_err =  state_est[4:7] - R_BO.apply([0, -omega_0, 0])
        h_w = state_est[7:]

        return np.concatenate((q_err_full[:3], omega_err, h_w)) 
    
    @abstractmethod
    def calc_input_cmds(self, att_state: np.ndarray, orbit_state: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Calculates the control input commands.

        Parameters
        ----------
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

class ZeroInputs(Controller):
    """
    A controller that always outputs zero for all control inputs.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def calc_input_cmds(self, att_state: np.ndarray, orbit_state: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Calculates the control input commands.

        Parameters
        ----------
        x : np.ndarray
            State vector.
        dt : float, optional
            Time step, by default None.

        Returns
        -------
        np.ndarray
            Control input vector (all zeros).
        """
        return np.zeros(6) 


class PI(Controller):
    """
    Proportional-Integral controller with anti-windup.
    """
    
    def __init__(self, K_q: np.ndarray, K_omega: np.ndarray, K_w: np.ndarray, K_q_int: np.ndarray, 
                 operating_point: Tuple[np.ndarray, np.ndarray], m: float|None = None, u_min: float|None = None, u_max: float|None = None) -> None:
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
        self.x_star, self.u_star = operating_point[0], operating_point[1]
        self.q_err_int = np.zeros(3)

        self.K_q = K_q
        self.K_omega = K_omega
        self.K_wheel = K_w
        self.K_q_int = K_q_int

        if m is None:
            m = 1.
        if u_min is None:
            u_min = -np.inf
        if u_max is None:
            u_max = np.inf
        self.u_min = u_min
        self.u_max = u_max
        self.m = m


    def calc_input_cmds(self, att_state: np.ndarray, orbit_state: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Calculates the control input.

        Parameters
        ----------
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
        D_x = self.calc_nadir_state_error(att_state, orbit_state) - self.x_star
        q_err = D_x[:3]
        omega_err = D_x[3:]
        h_w = D_x[6:]


        self.q_err_int += q_err * dt

        u_R = - (self.K_q @ q_err + self.K_omega @ omega_err + self.K_wheel @ h_w + self.K_q_int @ self.q_err_int)

        u = np.clip(u_R, self.u_min, self.u_max)

        self.q_err_int -= self.m * (u_R - u) # anti wind up

        return u + self.u_star
    
class LQR(PI):
    def __init__(self, f_x: Callable[[np.ndarray, np.ndarray], np.ndarray], f_u: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                 operating_point: Tuple[np.ndarray, np.ndarray], Q: np.ndarray, R: np.ndarray, 
                 m: float|None = None, u_min: float|None = None, u_max: float|None = None, K_q_int: np.ndarray|None = None) -> "PI":
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

        super().__init__(K[:, :3], K[:, 3:6], K[:, 6:9], K_q_int, operating_point, m, u_min, u_max)

class AvanziniLinear(PI):
    
    def __init__(self, k_p, k_d, k_i, k_m):
        """
        RW for attitude stabilization and Magnetorquers for momentum dumping
        
        x = [q_err, omega_err, h_w]
        u = [u_mag, u_rw]
        
        """
        K_q = np.eye(3) * k_p
        K_omega = np.eye(3) * k_d
        K_wheel = np.eye(3) * k_m
        K_q_int = np.eye(3) * k_i

        super().__init__(K_q, K_omega, K_wheel, K_q_int, (np.zeros(9), np.zeros(6))) 

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


    def calc_input_cmds(self, att_state: np.ndarray, orbit_state: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Calculates the control input using gain scheduling.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        dt : float, optional
            Time step (unused).

        Returns
        -------
        np.ndarray
            Control input vector.
        """
        x = self.calc_nadir_state_error(att_state, orbit_state)
        beta = self.calc_scheduling_param(x)

        i, j = self.closest_operating_points(beta)

        delta_u_i = np.dot(self.lqr_gains[i], ((w - self.operating_points[i][2]) - (x - self.operating_points[i][0])))
        delta_u_j = np.dot(self.lqr_gains[j], ((w - self.operating_points[j][2]) - (x - self.operating_points[j][0])))

        alpha = np.dot(self.rho[i] - self.rho[j], beta - self.rho[j]) / np.linalg.norm(self.rho[i] - self.rho[j])

        delta_u = delta_u_i * alpha + (1-alpha) * delta_u_j

        u_i = self.operating_points[i][1]
        
        u = delta_u + u_i

        return u
    
class FeedforwardController(Controller):

    def __init__(self, K_q: np.ndarray, K_omega: np.ndarray, J_hat: np.ndarray, B: np.ndarray, u_min: np.ndarray, u_max: np.ndarray):
        super().__init__()
        self.K_q = K_q
        self.K_omega = K_omega
        self.J_hat = J_hat
        self.B = B
        self.u_min = u_min
        self.u_max = u_max


    def calc_input_cmds(self, att_state: np.ndarray, orbit_state: np.ndarray, dt: float | None = None) -> np.ndarray:
        x = self.calc_input_cmds(att_state, orbit_state)
        q_err = x[:3]
        omega_err = x[3:]
        h_w = x[6:]

        omega = att_state[4:7]

        tau_cmd = np.cross(omega, self.J_hat @ omega + h_w) - (self.K_q @ q_err + self.K_omega @ omega_err)

        u = cgi_allocation(self.B, tau_cmd, self.u_min, self.u_max)

        return u
    

class MPC(Controller):
    def __init__(self, F: ca.Function, n_steps) -> None:
        super().__init__()

        self.F = F
        self.n_steps = n_steps

        self.X_last = None
        self.U_last = None

    def cost_function(self, X, U):
        #TODO: implement cost function

        # something like this:

        # cost = 0
        # for k in range(self.n_steps):
        #     cost += ca.mtimes([X[:, k].T, Q, X[:, k]])
        #     cost += ca.mtimes([U[:, k].T, R, U[:, k]])
        
        # cost += ca.mtimes([X[:, self.n_steps].T, Q, X[:, self.n_steps]])

        return cost

    def calc_input_cmds(self, att_state: np.ndarray, orbit_state: np.ndarray, dt: float | None = None) -> np.ndarray[Tuple[Any], np.dtype[np.float64]]:
        # TODO: See if optimal control problem can be turned into a ca.Function and compiled for computational speed

        x0 = self.calc_nadir_state_error(att_state, orbit_state)
        
        nx, nu = x0.size, 6

        opti = ca.Opti()

        X = opti.variable(nx, self.n_steps + 1)
        U = opti.variable(nu, self.n_steps)        

        #TODO: implement cost function
        opti.minimize(self.cost_function(...))

        # constraints
        opti.subject_to(X[:, 0] == x0)
        for k in range(self.n_steps):
            opti.subject_to(X[:, k+1] == self.F(X[:, k], U[:, k])) 

        # TODO: implement input constraints
        opti.subject_to(opti.bounded(...))
        opti.subject_to(opti.bounded(...))
        opti.subject_to(opti.bounded(...))

        if self.X_last is not None and self.U_last is not None:
            opti.set_initial(X[:, :-1], self.X_last[:, 1:])
            opti.set_initial(X[:, -1], self.X_last[:, -1])
            opti.set_initial(U[:, :-1], self.U_last[:, 1:])
            opti.set_initial(U[:, -1], self.U_last[:, -1]) 

        verbose = True
        opts = {
            'qpsol': 'qrqp',
            'print_header': verbose,
            'print_iteration': verbose,
            'print_time': verbose,
            'qpsol_options': {
                'print_iter': verbose,
                'print_header': verbose,
                'print_info': verbose,
                'max_iter': 100
            }
        }
        try:
            try:
                opti.solver('sqpmethod', opts)
                sol = opti.solve()
                
            except Exception as e:
                print(e)
                opti.solver('ipopt')

                sol = opti.solve()
        except Exception as e:
            self.debug(opti, X, U)    
            raise e

        self.X_last = sol.value(X)
        self.U_last = sol.value(U)

        #TODO: for logging maybe: sol.value(X), sol.value(U)

        return sol.value(U)[0]


    def debug(self, opti, X, U):

        X = opti.debug.value(X)
        U = opti.debug.value(U)

        print("-----------------------------------------")
        print(opti.debug)
        print("X")
        print(X[:3, :].T, "\n")


        print("U")
        print(opti.debug.value(U).T)

        print("Infeasibilites:")
        print(opti.debug.show_infeasibilities())


        # TODO: some kind of debugging plots
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        # ax1.plot(range(self.n_steps+1), X[:3, :].T)
        # ax2.plot(range(self.n_steps+1), obs_state_pred[:, :3])
        # ax3.plot(range(self.n_steps), U.T)

        # ax1.set_title("Position UAV")
        # ax2.set_title("Position Obstacle")
        # ax3.set_title("Inputs")
        # fig.tight_layout()
        # fig.savefig("debug_info.png")
        

        print("-----------------------------------------")
