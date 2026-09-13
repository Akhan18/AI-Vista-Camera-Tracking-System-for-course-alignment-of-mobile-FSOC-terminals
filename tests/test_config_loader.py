"""
test_config_loader.py

Basic tests for src/utils/config_loader.py.

Run with:
    python -m unittest tests/test_config_loader.py -v
(from the project root, AI-VISTA/)
"""

import unittest
from pathlib import Path

from src.utils.config_loader import load_config, merge_configs, load_scenario, Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class TestConfigLoader(unittest.TestCase):

    def test_default_config_loads(self):
        """The default config file should load without errors."""
        cfg = load_config(DEFAULT_CONFIG_PATH)
        self.assertIsInstance(cfg, Config)

    def test_required_sections_present(self):
        """All required top-level sections must exist after loading."""
        cfg = load_config(DEFAULT_CONFIG_PATH)
        for section in ["world", "camera", "beacon", "motion",
                         "disturbances", "performance_targets",
                         "simulation", "paths"]:
            self.assertTrue(hasattr(cfg, section), f"Missing section: {section}")

    def test_attribute_style_access(self):
        """Nested values should be reachable via dot notation."""
        cfg = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(cfg.world.width_px, 2000)
        self.assertEqual(cfg.world.height_px, 2000)
        self.assertEqual(cfg.camera.resolution.width_px, 640)
        self.assertEqual(cfg.camera.resolution.height_px, 480)
        self.assertEqual(cfg.beacon.size_px.width, 10)

    def test_performance_targets_match_problem_statement(self):
        """Sanity check that SIH target values were transcribed correctly."""
        cfg = load_config(DEFAULT_CONFIG_PATH)
        self.assertLessEqual(cfg.performance_targets.acquisition_time_s_max, 2.0)
        self.assertLessEqual(cfg.performance_targets.tracking_error_px_max, 10)
        self.assertGreaterEqual(cfg.performance_targets.processing_fps_min, 20)

    def test_missing_file_raises_filenotfounderror(self):
        """Loading a non-existent config file should fail clearly."""
        with self.assertRaises(FileNotFoundError):
            load_config(PROJECT_ROOT / "configs" / "does_not_exist.yaml")

    def test_merge_configs_overrides_only_specified_fields(self):
        """merge_configs should override nested fields without wiping siblings."""
        base = load_config(DEFAULT_CONFIG_PATH)
        override = Config({"motion": {"type": "circular"}})
        merged = merge_configs(base, override)

        # Overridden field changed
        self.assertEqual(merged.motion.type, "circular")
        # Sibling fields from base preserved
        self.assertEqual(merged.motion.circular.radius_px, 600)
        # Untouched sections preserved entirely
        self.assertEqual(merged.camera.resolution.width_px, 640)

    def test_load_scenario_merges_partial_override_file(self):
        """A partial scenario file (e.g. circular.yaml) should merge cleanly
        on top of default.yaml without needing every section repeated."""
        circular_path = PROJECT_ROOT / "configs" / "circular.yaml"
        cfg = load_scenario(DEFAULT_CONFIG_PATH, circular_path)

        self.assertEqual(cfg.motion.type, "circular")
        # Base sections still present and untouched
        self.assertEqual(cfg.world.width_px, 2000)
        self.assertEqual(cfg.camera.resolution.width_px, 640)


if __name__ == "__main__":
    unittest.main()
