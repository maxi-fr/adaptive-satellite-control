# TODO
from abc import ABC, abstractmethod
import datetime
from scipy.spatial.transform import Rotation as R
import numpy as np



# TODO: Refactor: only need 1 generic gaussian random walk bias + gaussian noise class. bias noise = 0 for sensor like SunSensor

class Sensor(ABC):
    def __init__(self, frequency: float):
        self.rng = np.random.default_rng()
        self.period = datetime.timedelta(seconds=1.0 / frequency)
        self.last_measurement = datetime.datetime(datetime.MINYEAR, 1, 1, tzinfo=datetime.timezone.utc)

    @abstractmethod
    def measure(self, t: datetime.datetime, *args, **kwargs):
        pass
        
    @abstractmethod
    def read(self, t: datetime.datetime):
        pass

class Accelerometer(Sensor):
    def __init__(self, frequency: float, sigma_sq: float, bias_sigma_sq: float):
        super().__init__(frequency)
        self.acc = np.zeros(3)
        self.bias = np.zeros(3)
        self.bias_sigma_sq = bias_sigma_sq
        self.sigma_sq = sigma_sq

    def measure(self, t: datetime.datetime, d_v: np.ndarray, R_BO: R): # type: ignore
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        dt = (t - self.last_measurement)
        if dt >= self.period:
            dt_s = dt.total_seconds()
            self.last_measurement = t

            bias = self.bias + np.sqrt(self.bias_sigma_sq) * np.sqrt(dt_s) * self.rng.normal(0, 1, 3)
            self.acc = (
                    d_v - R_BO.apply([0, 0, 9.81])
                    + 0.5 * (bias + self.bias)
                    + ((self.sigma_sq / dt_s + 1 / 12 * self.bias_sigma_sq * dt_s) ** 0.5)
                    * self.rng.normal(0, 1, 3)
            )
            self.bias = bias

    def read(self, t: datetime.datetime): # type: ignore
        """
        Returns last measured value for the accelerometer. 

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        np.ndarray, shape (3,)
            Specific force in the body frame [m/s^2].
        bool, 
            Returns true if the measurement is from the same times
        """

        return self.acc, self.last_measurement == t
    

class Gyroscope(Sensor):
    def __init__(self, frequency: float, sigma_sq: float, bias_sigma_sq: float):
        super().__init__(frequency)
        self.omega = np.zeros(3)
        self.bias = np.zeros(3)
        self.bias_sigma_sq = bias_sigma_sq
        self.sigma_sq = sigma_sq

    def measure(self, t: datetime.datetime, omega: np.ndarray): # type: ignore
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        dt = (t - self.last_measurement)
        if dt >= self.period:
            self.last_measurement = t
            dt_s = dt.total_seconds()

            bias = self.bias + np.sqrt(self.bias_sigma_sq) * np.sqrt(dt_s) * self.rng.normal(0, 1, 3)
            self.omega = (
                    omega
                    + 0.5 * (bias + self.bias)
                    + ((self.sigma_sq / dt_s + 1 / 12 * self.bias_sigma_sq * dt_s) ** 0.5)
                    * self.rng.normal(0, 1, 3)
            )
            self.bias = bias

    def read(self, t: datetime.datetime): # type: ignore
        """
        Returns last measured value for the gyroscope. 

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        np.ndarray, shape (3,)
            Angular velocity ration of the body frame with respect to the inertial frame in the body frame [rad/s].
        """

        return self.omega, self.last_measurement == t


class Magnetometer(Sensor):
    def __init__(self, frequency: float, sigma_sq: float, const_bias: np.ndarray):
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.const_bias = const_bias
        self.B = np.zeros(3)

    def measure(self, t: datetime.datetime, B_body: np.ndarray): # type: ignore
        dt = (t - self.last_measurement)

        if dt >= self.period:
            self.last_measurement = t
            self.B = self.const_bias + B_body + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)

    def read(self, t: datetime.datetime):
        """
        Returns last measured value for the magnetometer. 

        Parameters
        ----------
        t : datetime.datetime
            Current time.

        Returns
        -------
        np.ndarray, shape (3,)
            Earths magnetic field vector in the body frame [T].
        """
        return self.B, self.last_measurement == t
    

class SunSensor(Sensor):
    def __init__(self, frequency: float, sigma_sq: float):
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.sun_pos = np.zeros(3)

    def measure(self, t: datetime.datetime, sun_pos: np.ndarray):
        dt = (t - self.last_measurement)

        if dt >= self.period:
            self.last_measurement = t
            self.sun_pos = sun_pos + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)

    def read(self, t: datetime.datetime):

        return self.sun_pos, self.last_measurement == t
    

class GPS(Sensor):
    def __init__(self, frequency: float, sigma_sq: float):
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.sat_pos = np.zeros(3)


    def measure(self, t: datetime.datetime, sat_pos: np.ndarray): # type: ignore
        dt = (t - self.last_measurement)

        if dt >= self.period:
            self.last_measurement = t
            self.sat_pos = sat_pos + np.sqrt(self.sigma_sq) * self.rng.normal(0, 1, 3)

    def read(self, t: datetime.datetime): # type: ignore

        return self.sat_pos, self.last_measurement == t 
    

class RW_tachometer(Sensor):
    def __init__(self, frequency: float, sigma_sq: float):
        super().__init__(frequency)

        self.sigma_sq = sigma_sq
        self.omega = 0

    def measure(self, t: datetime.datetime, omega: np.ndarray): # type: ignore
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        dt = (t - self.last_measurement)
        if dt >= self.period:
            self.last_measurement = t
            self.omega = float(omega) + np.sqrt(self.sigma_sq) * float(self.rng.normal(0, 1))

    def read(self, t: datetime.datetime): # type: ignore

        return self.omega, self.last_measurement == t