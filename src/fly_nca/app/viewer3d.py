"""3D Colony & Connectome Viewer powered by FlyGym 2.1.1 / MuJoCo (NeuroMechFly v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fly_nca.body.mujoco_flygym import run_mujoco_colony_and_export_viewer


def export_colony_3d_viewer_html(
    output_path: Path | str = "colony_3d_viewer.html",
    *,
    n_founders: int = 4,
    frames: int = 14,
    seed: int = 108,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the 189,766-cell Fly-NCA colony in FlyGym/MuJoCo and export the 3-panel viewer."""
    return run_mujoco_colony_and_export_viewer(
        output_html_path=output_path,
        n_founders=n_founders,
        frames=frames,
        seed=seed,
        device=device,
    )
