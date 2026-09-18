import unittest

from amroute_core import Detections


class FakeClock:
    def __init__(self, now=1.0):
        self.now = now

    def __call__(self):
        return self.now


class DetectionsTest(unittest.TestCase):
    def test_requires_both_detectors(self):
        clock = FakeClock()
        state = Detections(5, clock=clock)
        state.siren_fire()
        self.assertIsNone(state.snapshot())

    def test_returns_stable_pair_identity(self):
        clock = FakeClock()
        state = Detections(5, clock=clock)
        state.siren_fire()
        clock.now = 3
        state.ambulance_fire()
        self.assertEqual((1, 1), state.snapshot())
        self.assertEqual((1, 1), state.snapshot())

    def test_rejects_events_too_far_apart(self):
        clock = FakeClock()
        state = Detections(5, clock=clock)
        state.siren_fire()
        clock.now = 7
        state.ambulance_fire()
        self.assertIsNone(state.snapshot())

    def test_pair_expires(self):
        clock = FakeClock()
        state = Detections(5, clock=clock)
        state.siren_fire()
        clock.now = 2
        state.ambulance_fire()
        clock.now = 8
        self.assertIsNone(state.snapshot())

    def test_window_must_be_positive(self):
        with self.assertRaises(ValueError):
            Detections(0)


if __name__ == "__main__":
    unittest.main()
