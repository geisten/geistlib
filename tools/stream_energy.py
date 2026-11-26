#!/usr/bin/env python3
"""tools/stream_energy.py — the power a host reports while tools/bench_stream
runs, laid over each row's timed frames (#533).

    stream_energy.py [--hz 5] [--idle 10] [--sampler-cpu N] [--gpu 0] -- \\
        bin/<target>/release/tools/bench_stream --window --camera 60 ...

The bench is started as a child with `--window` in its arguments, so that
it prints when each row's timed frames ran (CLOCK_MONOTONIC). This process
samples meanwhile, at `--hz` (at most 10 a second), from a thread of its own
that can be bound to a core the bench does not use (`--sampler-cpu`): no
sampling is in the bench's process, and none in its timed path. Before the
bench starts the same sources are sampled for `--idle` seconds, which is the
baseline every row is also given above.

What is read, where the host has it without privileges or equipment. Each
is a different part of the machine; none is the wall socket:

  gpu      `nvidia-smi --query-gpu=power.draw`: the graphics card as its
           driver reports it, the whole board (GPU, its memory, fans and
           regulators). Not the processor, not the rest of the machine.
  package  Linux powercap (RAPL), `/sys/class/powercap/intel-rapl:N/
           energy_uj`: the energy counter of each processor package, cores
           and what else is in the package. Not memory, not a graphics card,
           not the board. Since 2020 the file is readable by root alone on
           most systems; this tool says so and goes on without it.
  rails    Raspberry Pi 5, `vcgencmd pmic_read_adc`: current times voltage
           of every rail the power management chip measures both of, summed.
           That is what the chip delivers to the processor, the memory and
           the board's 3.3 V and 1.8 V parts. It leaves out the chip's own
           conversion loss and everything fed from 5 V directly: USB
           devices, the fan, an NVMe drive on the PCIe connector, a HAT.
  system   a Mac with a battery, `ioreg -r -c AppleSmartBattery`: the
           machine's power telemetry, `AccumulatedSystemLoad` over
           `SystemLoadAccumulatorCount`, the mean of what the whole machine
           draws behind its power adapter, display included. The registry
           has a new value about once a minute, so a row counts only the
           intervals that lie inside its window: a row has to run for two
           minutes and more (`--frames 7200` behind a camera at 60), and a
           window with no whole interval in it is "not read".

Per row: the mean power over the row's window and the energy per frame
(mean power times the window, by the frames the row computed), both as read
and above the idle baseline. A counter (package) is exact between two
readings; a power reading (gpu, rails) is a sample every 1/hz seconds on
average (at random intervals, or at a period that is no multiple of a
frame's, so that the samples do not all fall on the same point of a frame),
joined by straight lines; the Mac's telemetry (system) is a mean per
interval of about a minute.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time

WINDOW = re.compile(
    r"^\s*(\d+x\d+)\s+(\S+)\s+(\S+)\s+window ([0-9.]+) to ([0-9.]+) s on the monotonic clock, "
    r"(\d+) frames"
)


def now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def parse_pmic(text: str) -> float | None:
    """The sum of current times voltage over the rails of `vcgencmd
    pmic_read_adc` that have both, in watts."""
    amps: dict[str, float] = {}
    volts: dict[str, float] = {}
    for m in re.finditer(r"(\w+)_([AV]) \w+\(\d+\)=([0-9.]+)[AV]", text):
        (amps if m.group(2) == "A" else volts)[m.group(1)] = float(m.group(3))
    both = [amps[r] * volts[r] for r in amps if r in volts]
    return sum(both) if both else None


def mean_power(samples: list[tuple[float, float]], t0: float, t1: float) -> tuple[float, int] | None:
    """The mean of a sampled power over [t0, t1], the samples joined by
    straight lines and held flat before the first and behind the last; and
    how many samples lie inside."""
    if not samples or t1 <= t0:
        return None

    def at(t: float) -> float:
        if t <= samples[0][0]:
            return samples[0][1]
        for (ta, va), (tb, vb) in zip(samples, samples[1:]):
            if t <= tb:
                return va + (vb - va) * (t - ta) / (tb - ta) if tb > ta else vb
        return samples[-1][1]

    inside = [(t, v) for t, v in samples if t0 < t < t1]
    pts = [(t0, at(t0)), *inside, (t1, at(t1))]
    joules = sum((tb - ta) * (va + vb) / 2 for (ta, va), (tb, vb) in zip(pts, pts[1:]))
    return joules / (t1 - t0), len(inside)


def counter_power(samples: list[tuple[float, float]], t0: float, t1: float) -> tuple[float, int] | None:
    """The mean power between t0 and t1 from an energy counter in joules,
    read at the samples and joined by straight lines."""
    if len(samples) < 2 or t1 <= t0 or t0 < samples[0][0] or t1 > samples[-1][0]:
        return None

    def at(t: float) -> float:
        for (ta, va), (tb, vb) in zip(samples, samples[1:]):
            if t <= tb:
                return va + (vb - va) * (t - ta) / (tb - ta) if tb > ta else vb
        return samples[-1][1]

    return (at(t1) - at(t0)) / (t1 - t0), sum(1 for t, _ in samples if t0 < t < t1)


class Source:
    """One thing that is read: `samples` are (monotonic seconds, value)."""

    def __init__(self, name: str, what: str, counter: bool = False) -> None:
        self.name = name
        self.what = what
        self.counter = counter
        self.samples: list[tuple[float, float]] = []

    def power(self, t0: float, t1: float) -> tuple[float, int] | None:
        return (counter_power if self.counter else mean_power)(self.samples, t0, t1)

    def poll(self) -> None:  # called by the sampling thread, every tick
        pass

    def start(self, hz: float) -> None:  # called once, on the sampling thread
        pass

    def stop(self) -> None:
        pass


def parse_telemetry(text: str) -> tuple[int, int] | None:
    """(AccumulatedSystemLoad in mW, SystemLoadAccumulatorCount) out of
    `ioreg -rw0 -c AppleSmartBattery`."""
    acc = re.search(r'"AccumulatedSystemLoad"=(\d+)', text)
    count = re.search(r'"SystemLoadAccumulatorCount"=(\d+)', text)
    return (int(acc.group(1)), int(count.group(1))) if acc and count else None


def interval_power(
    updates: list[tuple[float, int, int]], t0: float, t1: float, slack: float
) -> tuple[float, int] | None:
    """The mean power in watts over the intervals between two updates (seen
    at, accumulated mW, count) that lie inside [t0 + slack, t1]: an update
    is seen up to `slack` seconds after the registry had it."""
    acc = count = n = 0
    for (ta, aa, ca), (tb, ab, cb) in zip(updates, updates[1:]):
        if ta - slack >= t0 and tb <= t1 and cb > ca:
            acc += ab - aa
            count += cb - ca
            n += 1
    return (acc / count * 1e-3, n) if count else None


class MacTelemetry(Source):
    PERIOD = 2.0  # seconds between two looks at the registry

    def __init__(self) -> None:
        super().__init__("system", "AppleSmartBattery power telemetry: the whole machine behind its adapter")
        self.updates: list[tuple[float, int, int]] = []
        self.looked = 0.0

    @staticmethod
    def read() -> tuple[int, int] | None:
        try:
            out = subprocess.run(
                ["ioreg", "-rw0", "-c", "AppleSmartBattery"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_telemetry(out)

    def poll(self) -> None:
        t = now()
        if t - self.looked < self.PERIOD:
            return
        self.looked = t
        v = self.read()
        if v is not None and (not self.updates or v[1] != self.updates[-1][2]):
            self.updates.append((now(), v[0], v[1]))

    def power(self, t0: float, t1: float) -> tuple[float, int] | None:
        return interval_power(self.updates, t0, t1, self.PERIOD)


class Rapl(Source):
    def __init__(self, paths: list[str]) -> None:
        super().__init__("package", "RAPL package energy, %d package(s)" % len(paths), counter=True)
        self.paths = paths
        self.range = [self._read(os.path.join(os.path.dirname(p), "max_energy_range_uj")) for p in paths]
        self.last = [self._read(p) for p in paths]
        self.total = 0.0

    @staticmethod
    def _read(path: str) -> int:
        with open(path) as f:
            return int(f.read())

    def poll(self) -> None:
        t = now()
        for i, p in enumerate(self.paths):
            v = self._read(p)
            self.total += ((v - self.last[i]) % self.range[i] if self.range[i] else v - self.last[i]) * 1e-6
            self.last[i] = v
        self.samples.append((t, self.total))


class Pmic(Source):
    def __init__(self) -> None:
        super().__init__("rails", "sum of current x voltage over the PMIC's rails (vcgencmd pmic_read_adc)")

    @staticmethod
    def read() -> float | None:
        try:
            out = subprocess.run(
                ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_pmic(out)

    def poll(self) -> None:
        t0 = now()
        w = self.read()
        if w is not None:
            self.samples.append(((t0 + now()) / 2, w))


class Nvidia(Source):
    def __init__(self, gpu: int) -> None:
        super().__init__("gpu", "nvidia-smi power.draw of GPU %d, the whole card" % gpu)
        self.gpu = gpu
        self.proc: subprocess.Popen[str] | None = None

    @staticmethod
    def period_ms(hz: float) -> int:
        """nvidia-smi loops at a fixed period: the next prime above 1000 / hz
        milliseconds, which is no multiple of a camera's frame period."""
        n = int(1000 / hz) + 1
        while any(n % d == 0 for d in range(2, int(n**0.5) + 1)):
            n += 1
        return n

    def start(self, hz: float) -> None:
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu),
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "-lms",
                str(self.period_ms(hz)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                self.samples.append((now(), float(line)))
            except ValueError:
                pass  # "[N/A]": the card does not say

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()


def find_sources(gpu: int, log) -> list[Source]:
    found: list[Source] = []
    if shutil.which("nvidia-smi"):
        found.append(Nvidia(gpu))
    else:
        log("gpu: no nvidia-smi here")
    paths = sorted(p for p in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
                   if re.search(r"intel-rapl:\d+/energy_uj$", p))
    if not paths:
        log("package: no /sys/class/powercap/intel-rapl:N here")
    else:
        try:
            found.append(Rapl(paths))
        except OSError as e:
            log("package: %s is not readable by this user (%s); not measured" % (paths[0], e.strerror))
    if shutil.which("vcgencmd") and Pmic.read() is not None:
        found.append(Pmic())
    else:
        log("rails: no vcgencmd pmic_read_adc here")
    if sys.platform == "darwin" and MacTelemetry.read() is not None:
        found.append(MacTelemetry())
    else:
        log("system: no AppleSmartBattery power telemetry here")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hz", type=float, default=5.0, help="samples a second, at most 10")
    ap.add_argument("--idle", type=float, default=10.0, help="seconds of baseline before the bench")
    ap.add_argument("--sampler-cpu", type=int, help="bind the sampling thread (and what it starts) to this core")
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi index of the card the bench runs on")
    ap.add_argument("--prefix", default="", help="put in front of every line this tool prints")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="-- bench_stream and its arguments")
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.hz <= 10:
        ap.error("a command behind --, and 0 < --hz <= 10")
    if "--window" not in command:
        command.append("--window")

    def log(text: str) -> None:
        print(args.prefix + "energy: " + text, flush=True)

    sources: list[Source] = []
    stop = threading.Event()
    ready = threading.Event()

    def sample() -> None:
        try:
            if args.sampler_cpu is not None and hasattr(os, "sched_setaffinity"):
                try:
                    os.sched_setaffinity(threading.get_native_id(), {args.sampler_cpu})
                except OSError as e:
                    log("--sampler-cpu %d: %s; the sampling thread is not bound" % (args.sampler_cpu, e.strerror))
            sources.extend(find_sources(args.gpu, log))
            for s in sources:
                s.start(args.hz)
            tick = now()
            while not stop.is_set():
                for s in sources:
                    s.poll()
                ready.set()  # behind the first reading: a counter has its start
                # Not in step with the frames: a reading every 200 ms is every
                # twelfth frame of a camera at 60, always at the same point of
                # a frame, and says whatever the pass does there.
                tick += random.uniform(0.5, 1.5) / args.hz
                stop.wait(max(0.0, tick - now()))
        finally:
            ready.set()  # whatever went wrong here, the bench runs

    thread = threading.Thread(target=sample, daemon=True)  # never keeps this process alive
    thread.start()
    ready.wait()
    for s in sources:
        log("%s = %s" % (s.name, s.what))
    if not sources:
        log("this host has nothing to read; the bench runs all the same")
    idle0 = now()
    time.sleep(args.idle if sources else 0)
    idle1 = now()

    windows: list[tuple[str, float, float, int]] = []
    bench = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    assert bench.stdout is not None
    for line in bench.stdout:
        m = WINDOW.match(line)
        if m:
            windows.append((" ".join(m.group(1, 2, 3)), float(m.group(4)), float(m.group(5)), int(m.group(6))))
        else:
            print(args.prefix + line, end="", flush=True)
    rc = bench.wait()
    time.sleep(2.0 / args.hz)  # the samples behind the last window
    stop.set()
    thread.join()
    for s in sources:
        s.stop()

    idle = {s.name: s.power(idle0, idle1) for s in sources}
    for s in sources:
        if idle[s.name] is not None:
            log("idle %s %.2f W over %.1f s before the bench (%d readings)"
                % (s.name, idle[s.name][0], idle1 - idle0, idle[s.name][1]))
    for row, t0, t1, frames in windows:
        parts = []
        for s in sources:
            p = s.power(t0, t1)
            if p is None or frames == 0:
                parts.append("%s not read" % s.name)
                continue
            watts, n = p
            text = "%s %.2f W, %.4f J/frame" % (s.name, watts, watts * (t1 - t0) / frames)
            if idle[s.name] is not None:
                above = watts - idle[s.name][0]
                text += " (above idle %.2f W, %.4f J/frame)" % (above, above * (t1 - t0) / frames)
            parts.append(text + ", %d %s" % (n, "intervals" if s.name == "system" else "samples"))
        log("%s  %.2f s, %d frames: %s" % (row, t1 - t0, frames, "; ".join(parts) or "nothing read"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
