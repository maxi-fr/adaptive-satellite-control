import numpy as np
import plotly.graph_objects as go

# ----------------------------
# Example satellite trajectory
# Replace with your real ECI arrays
# ----------------------------
R_orbit = 7000  # km
theta = np.linspace(0, 2*np.pi, 300)
x_sat = R_orbit * np.cos(theta)
y_sat = R_orbit * np.sin(theta)
z_sat = np.zeros_like(theta)

# ----------------------------
# Earth sphere
# ----------------------------
R_earth = 6371  # km

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_earth = R_earth * np.outer(np.cos(u), np.sin(v))
y_earth = R_earth * np.outer(np.sin(u), np.sin(v))
z_earth = R_earth * np.outer(np.ones_like(u), np.cos(v))

# ----------------------------
# Create Plotly figure
# ----------------------------
fig = go.Figure()

# Earth surface
fig.add_trace(go.Surface(
    x=x_earth,
    y=y_earth,
    z=z_earth,
    colorscale=[[0, "rgb(40,80,150)"], [1, "rgb(40,80,150)"]],
    showscale=False,
    opacity=1.0,
))

# Orbit path
fig.add_trace(go.Scatter3d(
    x=x_sat,
    y=y_sat,
    z=z_sat,
    mode="lines",
    line=dict(color="orange", width=6),
    name="Orbit"
))

# Current satellite position
fig.add_trace(go.Scatter3d(
    x=[x_sat[-1]],
    y=[y_sat[-1]],
    z=[z_sat[-1]],
    mode="markers",
    marker=dict(size=6, color="blue"),
    name="Satellite"
))

# ----------------------------
# Layout options
# ----------------------------
fig.update_layout(
    scene=dict(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)",
        aspectmode="data",   # equal aspect ratio
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2)  # camera position
        )
    ),
    title="Satellite Orbit in ECI Coordinates"
)

fig.show()