import unittest

from src.camera.controller import compute_error, error_to_command, step_control
from src.camera.virtual_camera import VirtualCamera


class TestControllerMath(unittest.TestCase):

    def test_compute_error_zero_when_centered(self):
        ex, ey = compute_error((320, 240), (320, 240))
        self.assertEqual((ex, ey), (0, 0))

    def test_compute_error_sign_and_magnitude(self):
        # Target to the right and below image centre -> positive errors.
        ex, ey = compute_error((350, 260), (320, 240))
        self.assertEqual(ex, 30)
        self.assertEqual(ey, 20)

    def test_error_to_command_center_when_small(self):
        self.assertEqual(error_to_command(0, 0), "CENTER")
        self.assertEqual(error_to_command(1, 1), "CENTER")  # below LOCK_THRESHOLD_PX

    def test_error_to_command_directions(self):
        self.assertEqual(error_to_command(50, 0), "RIGHT")
        self.assertEqual(error_to_command(-50, 0), "LEFT")
        self.assertEqual(error_to_command(0, 50), "DOWN")
        self.assertEqual(error_to_command(0, -50), "UP")

    def test_step_control_moves_camera_toward_target(self):
        camera = VirtualCamera(
            world_width=2000, world_height=2000, res_width=640, res_height=480,
            max_angular_speed_deg_per_s=5.0, fov_h_deg=4.0, fov_v_deg=3.0,
        )
        start_x, start_y = camera.center_x, camera.center_y

        # Target appears to the right of image centre in the camera frame.
        target_frame_xy = (400, 240)  # image centre is (320, 240)
        result = step_control(camera, target_frame_xy, dt=0.033, gain=0.5)

        self.assertEqual(result.command, "RIGHT")
        self.assertGreater(result.error_magnitude_px, 0)
        # Camera should have moved in the positive x direction (toward target).
        self.assertGreater(camera.center_x, start_x)
        self.assertEqual(camera.center_y, start_y)  # no y error, no y movement

    def test_step_control_respects_max_speed_clamp(self):
        camera = VirtualCamera(
            world_width=2000, world_height=2000, res_width=640, res_height=480,
            max_angular_speed_deg_per_s=5.0, fov_h_deg=4.0, fov_v_deg=3.0,
        )
        # A huge error (target far outside the frame, hypothetically).
        target_frame_xy = (10000, 240)
        step_control(camera, target_frame_xy, dt=0.033, gain=1.0)

        max_allowed_step = camera.max_speed_px_per_s_x * 0.033
        actual_step = camera.center_x - 1000  # camera started at world centre (1000)
        self.assertLessEqual(actual_step, max_allowed_step + 1e-6)


if __name__ == "__main__":
    unittest.main()
