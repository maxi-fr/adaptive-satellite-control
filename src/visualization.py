import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils import Surface


class SatelliteVisualizer:
    def __init__(self, surfaces):
        plt.ion()

        # self.fig_orbit = plt.figure(figsize=(7, 7))
        # self.ax_orbit = self.fig_orbit.add_subplot(111, projection="3d")
        # self.fig_orbit.add_axes(self.ax_orbit)
        # self.ax_orbit.set_title("Orbit")
        # self.ax_orbit.set_xlabel("X [m]")
        # self.ax_orbit.set_ylabel("Y [m]")
        # self.ax_orbit.set_zlabel("Z [m]")
        # self.ax_orbit.set_box_aspect([1, 1, 1])

        # # ---- Earth (draw ONCE) ----
        # self._draw_earth(self.ax_orbit, earth_radius)

        # # ---- Orbit artists (draw AFTER Earth) ----
        # self.tail_line, = self.ax_orbit.plot(
        #     [], [], [], color="orange", lw=2, zorder=1.2
        # )
        # self.sat_point, = self.ax_orbit.plot(
        #     [], [], [], "ro", markersize=5, zorder=1.3
        # )

        self.fig_att = plt.figure(figsize=(6, 6))
        self.ax_att = self.fig_att.add_subplot(111, projection="3d")
        self.ax_att.set_title("Attitude")
        self.ax_att.set_xlabel("X [m]")
        self.ax_att.set_ylabel("Y [m]")
        self.ax_att.set_zlabel("Z [m]")
        self.ax_att.set_box_aspect([1, 1, 1])

        L = max(
            np.linalg.norm(surf.x_axis) + np.linalg.norm(surf.y_axis)
            for surf in surfaces
        )
        self.ax_att.set_xlim(-L, L)
        self.ax_att.set_ylim(-L, L)
        self.ax_att.set_zlim(-L, L)
        self.L = L



        self.surface_patches = []
        self.vel_quiver = self.ax_att.quiver(0.0, 0.0, 0.0, 1, 1, 1, length=0.8 * L, color="blue", linewidth=2, label="Velocity") #type: ignore

        bbox = self.ax_att.get_position()

        w = 0.13
        h = 0.13
        dx = 0.0

        self.ax_frame_orbit = self.fig_att.add_axes(
            [bbox.x0 + dx, bbox.y1 - h - 0.05, w, h], #type: ignore
            projection="3d",
        )

        self.ax_frame_inertial = self.fig_att.add_axes(
            [bbox.x0 + dx + w + dx, bbox.y1 - h - 0.05, w, h], #type: ignore
            projection="3d",
        )

        self.ax_frame_body = self.fig_att.add_axes(
            [bbox.x0 + dx + 2 * (w + dx), bbox.y1 - h - 0.05, w, h], #type: ignore
            projection="3d",
        )




    # def _draw_earth(self, ax, R_e):
    #     u = np.linspace(0, 2 * np.pi, 40)
    #     v = np.linspace(0, np.pi, 20)

    #     x = R_e * np.outer(np.cos(u), np.sin(v))
    #     y = R_e * np.outer(np.sin(u), np.sin(v))
    #     z = R_e * np.outer(np.ones_like(u), np.cos(v))

    #     ax.plot_surface(
    #         x, y, z,
    #         color="blue",
    #         alpha=0.25,
    #         linewidth=0,
    #         antialiased=False,
    #         shade=True,
    #         zorder=1.1
    #     )

    def _draw_surfaces_attitude(self, surfaces, R_BO):
        for p in self.surface_patches:
            p.remove()
        self.surface_patches.clear()

        for surf in surfaces:
            p = surf.pos
            corners_b = np.array([
                p,
                p + surf.x_axis,
                p + surf.x_axis + surf.y_axis,
                p + surf.y_axis,
            ])

            corners_o = R_BO.inv().apply(corners_b)

            poly = Poly3DCollection(
                [corners_o],
                facecolor="cyan",
                edgecolor="k",
                alpha=0.6,
            )

            self.ax_att.add_collection3d(poly)
            self.surface_patches.append(poly)


    def _update_quiver(self, quiver, vec, color, label, scale):
        v = vec / (np.linalg.norm(vec) + 1e-12)

        quiver.remove()
        quiver = self.ax_att.quiver(0.0, 0.0, 0.0, v[0], v[1], v[2], length=scale, color=color, linewidth=2, label=label)

        return quiver
    
    def _draw_triad(self, ax, R_FO, label, length=1.0):
        """
        Draw a tight coordinate triad for frame F expressed in orbit frame.
        """
        ax.cla()

        e = np.eye(3)
        colors = ["r", "g", "b"]

        for i in range(3):
            v = R_FO.apply(e[i])
            ax.quiver(
                0, 0, 0,
                v[0], v[1], v[2],
                color=colors[i],
                linewidth=2,
                length=length,
                normalize=True,
            )

        lim = 1.  
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        ax.set_box_aspect([1, 1, 1])

        ax.margins(0)
        ax.set_proj_type("ortho")  
        ax.dist = 1               

        ax.set_title(label, fontsize=8, pad=2)
        ax.axis("off")

    
    def _update_reference_frames(self, R_OI, R_OB):
        self._draw_triad(
            self.ax_frame_orbit,
            R.identity(),
            "Orbit",
        )

        self._draw_triad(
            self.ax_frame_inertial,
            R_OI,
            "Inertial",
        )

        self._draw_triad(
            self.ax_frame_body,
            R_OB,
            "Body",
        )


    def update(self, surfaces, R_BO, R_OI, v_orc):
        # self.tail.append(r_eci)
        # tail = np.asarray(self.tail)

        # self.tail_line.set_data(tail[:, 0], tail[:, 1])
        # self.tail_line.set_3d_properties(tail[:, 2])

        # self.sat_point.set_data([r_eci[0]], [r_eci[1]])
        # self.sat_point.set_3d_properties([r_eci[2]])

        # lim = np.max(np.linalg.norm(tail, axis=1)) * 1.1
        # self.ax_orbit.set_xlim(-lim, lim)
        # self.ax_orbit.set_ylim(-lim, lim)
        # self.ax_orbit.set_zlim(-lim, lim)

        self._draw_surfaces_attitude(surfaces, R_BO)

        self.vel_quiver = self._update_quiver(self.vel_quiver, v_orc, "blue", "Velocity", scale=0.8 * self.L)

        self._update_reference_frames(R_OI, R_BO.inv())

        plt.pause(0.01)


# if __name__ == "__main__":
#     import time

#     L = 1.0
#     R_id = np.eye(3)

#     surfaces = [
#         Surface(np.array([-0.5, -0.5,  0.5]), L, L, R_id),
#         Surface(np.array([-0.5, -0.5, -0.5]), L, L, np.diag([1, -1, -1])),
#     ]

#     viz = SatelliteVisualizer()

#     R_orbit = 7000e3
#     omega = 2 * np.pi / (90 * 60)

#     for k in range(-500, 500):
#         t = 4.0 * k
#         r = R_orbit * np.array([np.cos(omega*t), np.sin(omega*t), 0])
#         v = np.zeros(3)
#         R_BO = R.from_euler("z", 0.02 * t)

#         viz.update(surfaces, r, v, R_BO)
#         time.sleep(0.002)
