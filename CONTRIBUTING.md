# Contributing to Am-Rout

Thanks for helping improve the prototype. Am-Rout touches machine learning, traffic
simulation, and safety-sensitive ideas, so contributions should be small, testable,
and explicit about their assumptions.

## Before starting

1. Open an issue for new features, model changes, traffic-control changes, or research
   claims. Describe the problem and how you plan to validate the result.
2. Keep model weights and datasets out of Git unless their license and redistribution
   terms have been reviewed by the maintainer.
3. Do not present simulated results as evidence of real-world safety or effectiveness.

## Local checks

Create the environment with `uv sync`, then run:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

If your change affects live detection, also run the relevant setup and hardware checks:

```bash
uv run python scripts/check_setup.py --mode detection
uv run python detection.py --audio-only
uv run python detection.py --vision-only --source path/to/test-video.mp4
```

If your change affects the simulation, document the SUMO version, command, seed,
congestion level, and detection radius used to validate it. Reuse paired seeds when
comparing interventions.

## Pull requests

- Explain what changed, why, and how it was tested.
- Include focused tests for behavior that does not require hardware.
- Keep generated networks, model weights, recordings, videos, and result CSVs out of
  the commit.
- Update the README or `sim/SPEC.md` when commands, assumptions, or results change.

## Reporting security or safety concerns

Do not publish an exploit or a safety-critical failure path before contacting the
maintainer privately. Ordinary bugs and reproducibility gaps can use GitHub Issues.
