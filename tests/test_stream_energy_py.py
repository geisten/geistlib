#!/usr/bin/env python3
"""Hermetic checks for tools/stream_energy.py: what it reads out of the
bench's and the board's lines, and how samples become a mean power."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stream_energy", ROOT / "tools/stream_energy.py")
energy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(energy)


class StreamEnergyTest(unittest.TestCase):
    def test_window_line_of_the_bench(self):
        m = energy.WINDOW.match(
            "1920x1080  rgb8  stream  window 1234.500000 to 1244.500000 s on the monotonic "
            "clock, 600 frames\n")
        self.assertEqual(m.group(1, 2, 3, 6), ("1920x1080", "rgb8", "stream", "600"))
        self.assertEqual(float(m.group(5)) - float(m.group(4)), 10.0)
        self.assertIsNone(energy.WINDOW.match("1920x1080  rgb8  stream  camera at 60.0 frames/s"))

    def test_rails_with_current_and_voltage_are_summed(self):
        text = ("   3V3_SYS_A current(1)=0.50000000A\n  VDD_CORE_A current(7)=2.00000000A\n"
                "   3V3_SYS_V volt(9)=3.30000000V\n  VDD_CORE_V volt(15)=0.90000000V\n"
                "     EXT5V_V volt(24)=5.10000000V\n")
        self.assertAlmostEqual(energy.parse_pmic(text), 0.5 * 3.3 + 2.0 * 0.9)
        self.assertIsNone(energy.parse_pmic("nothing"))

    def test_mean_of_sampled_power(self):
        ramp = [(0.0, 0.0), (1.0, 10.0), (2.0, 20.0)]
        self.assertAlmostEqual(energy.mean_power(ramp, 0.0, 2.0)[0], 10.0)
        self.assertAlmostEqual(energy.mean_power(ramp, 0.5, 1.5)[0], 10.0)
        self.assertEqual(energy.mean_power(ramp, 0.5, 1.5)[1], 1)
        # flat before the first sample and behind the last
        self.assertAlmostEqual(energy.mean_power(ramp, 2.0, 4.0)[0], 20.0)
        self.assertIsNone(energy.mean_power([], 0.0, 1.0))

    def test_the_gpu_is_not_read_in_step_with_the_frames(self):
        self.assertEqual(energy.Nvidia.period_ms(5), 211)
        self.assertEqual(energy.Nvidia.period_ms(10), 101)

    def test_mean_power_of_a_counter(self):
        joules = [(0.0, 100.0), (1.0, 130.0), (2.0, 190.0)]
        self.assertAlmostEqual(energy.counter_power(joules, 0.0, 2.0)[0], 45.0)
        self.assertAlmostEqual(energy.counter_power(joules, 0.5, 1.5)[0], 45.0)
        self.assertIsNone(energy.counter_power(joules, 1.0, 3.0))  # not read to its end

    def test_the_macs_telemetry_counts_whole_intervals_inside_the_window(self):
        text = '"PowerTelemetryData" = {"SystemLoad"=70149,"AccumulatedSystemLoad"=1000000,' \
               '"SystemLoadAccumulatorCount"=100}'
        self.assertEqual(energy.parse_telemetry(text), (1000000, 100))
        self.assertIsNone(energy.parse_telemetry("no such class"))
        # seen at 10, 70, 130, 190 s: 55 readings each, of 20 W, 30 W and 50 W
        ups = [(10.0, 0, 0), (70.0, 55 * 20000, 55), (130.0, 55 * 50000, 110), (190.0, 55 * 100000, 165)]
        self.assertEqual(energy.interval_power(ups, 60.0, 200.0, 2.0), (40.0, 2))
        self.assertEqual(energy.interval_power(ups, 0.0, 135.0, 2.0), (25.0, 2))
        # the interval that began before the window (or within the slack) is not counted
        self.assertEqual(energy.interval_power(ups, 9.0, 135.0, 2.0), (30.0, 1))
        self.assertIsNone(energy.interval_power(ups, 71.0, 129.0, 2.0))


if __name__ == "__main__":
    unittest.main()
