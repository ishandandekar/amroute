import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_setup.py"
SPEC = importlib.util.spec_from_file_location("check_setup", SCRIPT)
check_setup = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check_setup)


class SetupCheckTest(unittest.TestCase):
    def test_missing_models_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = check_setup.check_detection(Path(tmp))
        model_results = [passed for passed, label in results if label.startswith("Model file:")]
        self.assertEqual([False, False], model_results)

    def test_model_files_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "brite").mkdir()
            (root / "sirenn").mkdir()
            (root / "brite" / "best_YOLO_ambulance_detect.pt").touch()
            (root / "sirenn" / "sireNN.pt").touch()
            results = check_setup.check_detection(root)
        model_results = [passed for passed, label in results if label.startswith("Model file:")]
        self.assertEqual([True, True], model_results)


if __name__ == "__main__":
    unittest.main()
