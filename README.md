# Am-Rout

Am-Rout is a research prototype for measuring how early ambulance detection could
reduce emergency-vehicle travel time in congested traffic. It combines:

- **sireNN** — local, real-time ambulance-siren classification;
- **brite** — YOLO-based visual ambulance detection;
- **sensor fusion** — confirmation when audio and vision events occur together; and
- **SUMO simulation** — a reproducible green-corridor experiment using traffic-light
  preemption and lane clearance.

> [!WARNING]
> Am-Rout is experimental software. It has not been validated for emergency dispatch,
> traffic-signal control, or any other safety-critical deployment.

## Repository map

| Path | Purpose |
| --- | --- |
| `detection.py` | Runs audio and vision detectors together or independently |
| `sirenn/` | Audio preprocessing, LSTM training, and live/file inference |
| `brite/` | YOLO video inference helpers |
| `sim/` | SUMO scenario generation, corridor control, batch runs, and analysis |
| `sim/SPEC.md` | Simulation design, assumptions, milestones, and current findings |
| `scripts/check_setup.py` | Reports missing dependencies, binaries, and model files |

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A webcam and microphone for live fused detection
- PortAudio development/runtime libraries for PyAudio
- Linux plus [Eclipse SUMO](https://sumo.dlr.de/docs/Installing/index.html) for the
  simulation workflow

On Debian or Ubuntu, install the native prerequisites with:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev sumo sumo-tools
```

Then create the project environment:

```bash
git clone https://github.com/ishandandekar/amroute.git
cd amroute
uv sync
uv run python scripts/check_setup.py --mode detection
```

The setup check intentionally exits with a non-zero status until all required model
files are present.

## Model files

Model weights and training datasets are not committed to Git. Before running the
detector, provide:

| Component | Default location |
| --- | --- |
| YOLO ambulance detector | `brite/best_YOLO_ambulance_detect.pt` |
| sireNN audio classifier | `sirenn/sireNN.pt` |

You can also keep weights elsewhere and pass `--model` and `--audio-model`. Only use
weights whose source and license you have verified.

To train sireNN, place WAV files in the class directories documented in
`sirenn/train_pytorch.py`, then run from that directory:

```bash
cd sirenn
uv sync
uv run python train_pytorch.py --model-path sireNN.pt
```

The repository does not yet include a reproducible training pipeline or distributable
weights for the YOLO model. That is a known project limitation, not an automatic
download performed by the code.

## Run detection

Fused audio and vision detection:

```bash
uv run python detection.py
```

Run only one sensor while developing or diagnosing hardware:

```bash
uv run python detection.py --audio-only
uv run python detection.py --vision-only --source path/to/video.mp4
```

Useful options:

```bash
uv run python detection.py --list-devices
uv run python detection.py --help
```

Press `q` in the video window or `Ctrl+C` in the terminal to stop. Missing model
files are reported before camera or microphone access begins.

## Run the SUMO study

The complete experiment design and interpretation live in [`sim/SPEC.md`](sim/SPEC.md).
The committed `corridor.osm` is the source network; generated SUMO networks, routes,
and result CSVs are intentionally ignored.

On Linux, first build the network and scenarios:

```bash
netconvert \
  --osm-files sim/corridor/corridor.osm \
  --output-file sim/corridor/corridor.net.xml

export SUMO_TOOLS=/usr/share/sumo/tools
uv run python sim/scenario.py --all --seed 1
uv run python scripts/check_setup.py --mode simulation
```

Run one baseline/preemption pair:

```bash
uv run python sim/run.py baseline --density low --seed 1
uv run python sim/run.py preempt --density low --seed 1 --r 200
```

Run the full 150-cell experiment and regenerate the analysis:

```bash
uv run python sim/run.py batch --resume
uv run python sim/analysis/analyze.py
```

Use `--gui` on a baseline or preemption command to watch the ambulance in SUMO.

## Development

The lightweight checks do not download ML dependencies or model weights:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## Current limitations

- Model weights are not distributed, and the visual training process is not yet
  reproducible from this repository.
- The simulation workflow is Linux-oriented.
- Simulation output is generated locally; the written report should not be treated as
  independently reproduced without rerunning the matrix.
- Audio/vision accuracy has not been benchmarked against a published held-out dataset.
- The project does not currently include a license. The maintainer must choose one
  before outside contributors can safely reuse or redistribute the code.

## Contributing

Bug reports and focused improvements are welcome. Good first areas are reproducible
model training, detector benchmarks, tests around simulation control, and documentation.
Please coordinate large or safety-relevant changes in an issue first.
