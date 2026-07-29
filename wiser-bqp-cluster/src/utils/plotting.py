"""Plotting helpers for solution fields and training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from ..pdes.base import PDE, Model  # noqa: E402


@torch.no_grad()
def plot_solution(
    model: Model, pde: PDE, out_path: str | Path, nx: int = 256, nt: int = 100
) -> None:
    """Save a 3-panel figure: prediction, reference, and absolute error."""
    coords, X, T = pde.grid(nx, nt)
    pred = model(coords).reshape(T.shape)
    ref = pde.reference(coords).reshape(T.shape)
    err = (pred - ref).abs()

    extent = [pde.domain["t"][0], pde.domain["t"][1],
              pde.domain["x"][0], pde.domain["x"][1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(
        axes, [pred.T, ref.T, err.T], ["prediction", "reference", "|error|"]
    ):
        im = ax.imshow(
            field.cpu().numpy(), extent=extent, origin="lower",
            aspect="auto", cmap="viridis",
        )
        ax.set_title(f"{pde.name}: {title}")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_history(history: list[dict], out_path: str | Path) -> None:
    """Plot loss components (and rel-L2 if present) over epochs."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ("loss", "loss_pde", "loss_data"):
        ys = [(h["epoch"], h[key]) for h in history if key in h]
        if ys:
            ax.semilogy(*zip(*ys), label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()

    rel = [(h["epoch"], h["rel_l2"]) for h in history if "rel_l2" in h]
    if rel:
        ax2 = ax.twinx()
        ax2.semilogy(*zip(*rel), "k--", label="rel_l2")
        ax2.set_ylabel("rel L2")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
