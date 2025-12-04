import numpy as np
import pymsis
import pyIGRF
import datetime
import pymap3d
import astropy.coordinates as coord
from astropy import units as u
from astropy.time import Time



def atmosphere_density_static(altitude: float) -> float:
    """
    Calculate atmospheric density using a simple static exponential model.

    This model is based on tabulated values from "Fundamentals of Spacecraft
    Attitude Determination and Control" by F. Markley and John Crassidis,
    Table D.1. It is a first-order approximation valid for altitudes
    between 300 and 800 km.

    Parameters
    ----------
    altitude : float
        Altitude in kilometers [m].

    Returns
    -------
    float
        Atmospheric density in kg/m^3.

    Raises
    ------
    ValueError
        If the altitude is outside the valid range of 300 to 800 km.
    """
    altitude = altitude / 1000.0

    const = {"p_0":[2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14],
                "h_0":[300, 350, 400, 450, 500, 600, 700],
                "H":[52.5, 56.4, 59.4, 62.2, 65.8, 79, 109]}
    if altitude < 300 or altitude > 800:
        raise ValueError(f"Altitude {altitude} km is outside the valid range for the static atmospheric model (300-800 km).")
    else:
        # depending on the altitude a different set of constants needs to be used
        # the altitude should be bigger than h_0 but smaller than the next value of h_0
        i = [j > altitude for j in const["h_0"]].index(True) - 1
    
    return const["p_0"][i] * np.exp(-(altitude - const["h_0"][i])/const["H"][i])


def atmosphere_density_msis(dt_utc: datetime.datetime, lat_deg: float, lon_deg: float, alt_m: float, 
                       f107: float = 150, f107a: float = 150, ap: int = 4) -> float:
    """
    Calculate atmospheric density using the pymsis library.

    This function calls the MSIS model to get atmospheric density for a
    specific time and location.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the density calculation.
    lat_deg : float
        Latitude in degrees.
    lon_deg : float
        Longitude in degrees.
    alt_m : float
        Altitude in meters.
    f107 : float, optional
        Daily F10.7 solar flux, by default 150.
    f107a : float, optional
        81-day average of F10.7 solar flux, by default 150.
    ap : int, optional
        The Ap geomagnetic index, by default 4.

    Returns
    -------
    float
        The calculated total mass density in kg/m^3.

    """

    # datetimes = np.array([dt_utc], dtype="datetime64[s]")
    alt_km = alt_m / 1000.0

    result = pymsis.calculate(dt_utc, lat_deg, lon_deg, alt_km, f107, f107a, ap)

    rho_kg_m3 = result[:, 0].item(0)

    return rho_kg_m3


def magnetic_field_vector(dt_utc: datetime.datetime, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """
    Calculates the Earth's magnetic field vector in the ECI frame.

    This function uses the IGRF model to get the magnetic field components
    in the North-East-Down (NED) frame, then transforms them to the
    Earth-Centered Inertial (ECI) J2000 frame.


    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the calculation.
    lat_deg : float
        Geodetic latitude in degrees.
    lon_deg : float
        Geodetic longitude in degrees.
    alt_m : float
        Altitude above the WGS84 ellipsoid in meters.

    Returns
    -------
    np.ndarray, shape (3,)
        The magnetic field vector [Bx, By, Bz] in the ECI frame, in nanoTeslas [nT].
    """

    D, I, H, Bn, Be, Bv, B_tot = pyIGRF.igrf_value(lat_deg, lon_deg, alt_m/1000)

    B_ecef = pymap3d.ned2ecef(Bn, Be, Bv, lat_deg, lon_deg, alt_m, pymap3d.Ellipsoid.from_name("wgs84"))

    B_eci = np.asarray(pymap3d.ecef2eci(*B_ecef, time=dt_utc))

    return (B_eci / np.linalg.norm(B_eci)) * B_tot

#TODO: sun and moon position can be precomputed vectorized for time efficiency
def sun_position(dt_utc: datetime.datetime) -> np.ndarray:
    """
    Calculates the Sun's position vector in the GCRS (ECI) frame.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the calculation.

    Returns
    -------
    np.ndarray, shape (3,)
        The Sun's position vector [x, y, z] in the GCRS frame, in meters [m].
    """

    time = Time(dt_utc.strftime("%Y-%m-%d %H:%M:%S"), scale="utc")
    sun = coord.get_sun(time)

    return sun.cartesian.xyz.to(u.m).value # type: ignore

def moon_position(dt_utc: datetime.datetime) -> np.ndarray:
    """
    Calculates the Moon's position vector in the GCRS (ECI) frame.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the calculation.

    Returns
    -------
    np.ndarray, shape (3,)
        The Moon's position vector [x, y, z] in the GCRS frame, in meters [m].
    """

    time = Time(dt_utc.strftime("%Y-%m-%d %H:%M:%S"), scale="utc")
    moon = coord.get_body("moon", time)

    return moon.cartesian.xyz.to(u.m).value # type: ignore
    
def solar_radiation_pressure_constant(r_eci: np.ndarray, sun_pos_eci: np.ndarray) -> float:
    """
    Calculates the solar radiation pressure at the satellite's location.

    Parameters
    ----------
    r_eci : np.ndarray, shape (3,)
        Satellite position vector in the ECI frame [m].
    sun_pos_eci : np.ndarray, shape (3,)
        Sun position vector in the ECI frame [m].

    Returns
    -------
    float
        Solar radiation pressure [N/m^2].
    """

    P_1AU = 4.56e-6  # Solar radiation pressure at 1 AU [N/m^2]
    AU_M = 149597870700.0  # 1 Astronomical Unit in meters

    dist = np.asarray(sun_pos_eci) - np.asarray(r_eci)

    return float(P_1AU * (AU_M / np.linalg.norm(dist))**2) #FIXME: maybe wrong formula


E_RADIUS_M = 6378137.0 # Earth's equatorial radius in meters
def is_in_shadow(r_eci: np.ndarray, sun_pos_eci: np.ndarray) -> bool:
    """
    Determines if the satellite is in Earth's shadow using a cylindrical model.

    This model assumes the Sun's rays are parallel and casts a cylindrical
    shadow behind the Earth.

    Parameters
    ----------
    r_eci : np.ndarray, shape (3,)
        Satellite position vector in the ECI frame [m].
    sun_pos_eci : np.ndarray, shape (3,)
        Sun position vector in the ECI frame [m].

    Returns
    -------
    bool
        True if the satellite is in shadow, False otherwise.
    """
    
    sun_position_unit = sun_pos_eci / np.linalg.norm(sun_pos_eci)
    return np.dot(r_eci, sun_position_unit) < - np.sqrt(np.linalg.norm(r_eci)**2 - E_RADIUS_M**2)



if __name__ == "__main__":
    print(atmosphere_density_static(500))
    print(atmosphere_density_msis(datetime.datetime.now(), 40, -73, 500_000))

    magnetic_field_vector(datetime.datetime.now(), 40, -73, 500_000)

    sun_position(datetime.datetime.now())