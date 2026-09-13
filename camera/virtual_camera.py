"""
virtual_camera.py

Virtual camera model for the AI-VISTA FSOC simulation.

The camera has:
- a fixed image resolution,
- a configurable horizontal/vertical field of view,
- a movable optical-axis position in the 2D world,
- a maximum angular movement rate.

The camera converts angular offsets into image-plane pixel offsets.
"""

from dataclasses import dataclass


@dataclass
class VirtualCamera:
    world_width: int
    world_height: int
    res_width: int
    res_height: int

    max_angular_speed_deg_per_s: float
    fov_h_deg: float
    fov_v_deg: float

    initial_center_x: float
    initial_center_y: float

    def __post_init__(self):
        # Camera optical-axis position in the simulated world.
        self.center_x = self.initial_center_x
        self.center_y = self.initial_center_y

        # Pixels corresponding to one degree of angular displacement.
        self.pixels_per_degree_x = self.res_width / self.fov_h_deg
        self.pixels_per_degree_y = self.res_height / self.fov_v_deg

        # Maximum camera movement expressed as world-pixel equivalent.
        #
        # This is still a 2D approximation because the simulation does not
        # currently model target range/depth.
        self.max_speed_px_per_s_x = (
            self.max_angular_speed_deg_per_s
            * self.pixels_per_degree_x
        )

        self.max_speed_px_per_s_y = (
            self.max_angular_speed_deg_per_s
            * self.pixels_per_degree_y
        )

    @classmethod
    def from_config(cls, cfg) -> "VirtualCamera":
        return cls(
            world_width=cfg.world.width_px,
            world_height=cfg.world.height_px,
            res_width=cfg.camera.resolution.width_px,
            res_height=cfg.camera.resolution.height_px,
            max_angular_speed_deg_per_s=(
                cfg.camera.max_angular_speed_deg_per_s
            ),
            fov_h_deg=cfg.camera.field_of_view_deg.horizontal,
            fov_v_deg=cfg.camera.field_of_view_deg.vertical,
            initial_center_x=cfg.camera.initial_center_xy[0],
            initial_center_y=cfg.camera.initial_center_xy[1],
        )

    def viewport_bounds(self) -> tuple[int, int, int, int]:
        """
        Return the world region currently covered by the camera.

        The viewport remains useful for determining whether a beacon is
        inside the camera's field of view.
        """

        half_w = self.res_width / 2
        half_h = self.res_height / 2

        x_min = self.center_x - half_w
        y_min = self.center_y - half_h

        x_min = min(
            max(x_min, 0),
            self.world_width - self.res_width,
        )

        y_min = min(
            max(y_min, 0),
            self.world_height - self.res_height,
        )

        x_max = x_min + self.res_width
        y_max = y_min + self.res_height

        return (
            int(x_min),
            int(y_min),
            int(x_max),
            int(y_max),
        )

    def world_to_frame(
        self,
        world_x: float,
        world_y: float,
    ) -> tuple[float, float]:
        """
        Convert a world position to camera-frame pixel coordinates.

        The current 2D model treats the camera optical axis as the image
        centre and uses the configured FOV to establish the angular
        scale of the image.
        """

        dx_world = world_x - self.center_x
        dy_world = world_y - self.center_y

        # Convert world displacement to an angular displacement.
        #
        # Because this simulation has no range/depth model, world pixels
        # are treated as the angular-coordinate representation.
        angle_x_deg = dx_world / self.pixels_per_degree_x
        angle_y_deg = dy_world / self.pixels_per_degree_y

        # Convert angular displacement back onto the image plane.
        frame_x = (
            self.res_width / 2
            + angle_x_deg * self.pixels_per_degree_x
        )

        frame_y = (
            self.res_height / 2
            + angle_y_deg * self.pixels_per_degree_y
        )

        return frame_x, frame_y

    def move_by(
        self,
        dx: float,
        dy: float,
        dt: float,
    ):
        """
        Move the camera optical axis.

        Movement is limited by the configured maximum angular speed.
        """

        max_dx = self.max_speed_px_per_s_x * dt
        max_dy = self.max_speed_px_per_s_y * dt

        dx = max(-max_dx, min(dx, max_dx))
        dy = max(-max_dy, min(dy, max_dy))

        self.center_x += dx
        self.center_y += dy

        self.center_x = min(
            max(self.center_x, 0),
            self.world_width,
        )

        self.center_y = min(
            max(self.center_y, 0),
            self.world_height,
        )

    def image_center(self) -> tuple[float, float]:
        """Return the optical-axis position in image coordinates."""

        return (
            self.res_width / 2,
            self.res_height / 2,
        )