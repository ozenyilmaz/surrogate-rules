"""
tests/unit/test_loaders.py
--------------------------
Unit tests for the data loading layer.
"""

import numpy as np
import pytest

from src.data.loaders import normalize_labels, register_loader, load_dataset, LOADER_REGISTRY


# ─────────────────────────────────────────────
# normalize_labels
# ─────────────────────────────────────────────

class TestNormalizeLabels:

    def test_already_minus_one_plus_one(self):
        y = np.array([-1, 1, -1, 1])
        result = normalize_labels(y)
        np.testing.assert_array_equal(result, y)

    def test_zero_one_maps_correctly(self):
        y = np.array([0, 1, 0, 1])
        result = normalize_labels(y)
        np.testing.assert_array_equal(result, [-1, 1, -1, 1])

    def test_one_two_maps_correctly(self):
        y = np.array([1, 2, 1, 2])
        result = normalize_labels(y)
        np.testing.assert_array_equal(result, [-1, 1, -1, 1])

    def test_unsupported_label_set_raises(self):
        y = np.array([0, 2, 3])
        with pytest.raises(ValueError, match="Unsupported label set"):
            normalize_labels(y)

    def test_returns_int_array(self):
        y = np.array([0, 1])
        result = normalize_labels(y)
        assert result.dtype == int


# ─────────────────────────────────────────────
# LOADER_REGISTRY
# ─────────────────────────────────────────────

class TestLoaderRegistry:

    def test_known_datasets_are_registered(self):
        for name in ["mushroom", "banknote", "diabetes", "ionosphere"]:
            assert name in LOADER_REGISTRY, f"'{name}' not in registry"

    def test_register_custom_loader(self):
        def _dummy_loader(df):
            return np.zeros((3, 2)), np.array([0, 1, 0])

        register_loader("my_custom", _dummy_loader)
        assert "my_custom" in LOADER_REGISTRY

    def test_unknown_dataset_raises(self, tmp_path):
        # Create a minimal .arff file with an unknown name
        arff_content = (
            "@relation unknown\n"
            "@attribute x NUMERIC\n"
            "@attribute class {0,1}\n"
            "@data\n"
            "1.0,0\n"
        )
        p = tmp_path / "unknown_dataset.arff"
        p.write_text(arff_content)
        with pytest.raises(ValueError, match="No loader registered"):
            load_dataset(p)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/data.arff")
