"""SAI CLI: sai detect, sai eval, sai calibrate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from sai.io import load_image
from sai.pipeline import DetectorPipeline
from sai.eval import evaluate, cross_generator_eval

app = typer.Typer(help="SAI: Synthetic AI Image detector")
console = Console()


@app.command()
def detect(
    image_path: Path = typer.Argument(..., exists=True, help="Path to image file"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of rich text"),
) -> None:
    """Detect whether an image is AI-generated."""
    img = load_image(image_path)
    pipeline = DetectorPipeline()
    res = pipeline.detect(img)
    if json_out:
        console.print_json(json.dumps(res.to_dict()))
        return
    table = Table(title=f"SAI detection: {image_path.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Verdict", res.verdict.upper())
    table.add_row("Raw score", f"{res.raw_score:.4f}")
    table.add_row("Calibrated score", f"{res.calibrated_score:.4f}")
    table.add_row("Epistemic uncertainty", f"{res.epistemic_uncertainty:.4f}")
    table.add_row("Aleatoric uncertainty", f"{res.aleatoric_uncertainty:.4f}")
    table.add_row("Total uncertainty", f"{res.total_uncertainty:.4f}")
    console.print(table)

    sig_table = Table(title="Signals")
    sig_table.add_column("Signal", style="cyan")
    sig_table.add_column("Score", style="magenta")
    sig_table.add_column("Weight", style="green")
    for s, r in zip(pipeline.signals, res.signal_results):
        sig_table.add_row(s.name, f"{r.score:.4f}", f"{r.weight:.4f}")
    console.print(sig_table)


@app.command()
def eval_dir(
    real_dir: Path = typer.Argument(..., exists=True, help="Directory of real images"),
    ai_dir: Path = typer.Argument(..., exists=True, help="Directory of AI images"),
    generator: str = typer.Option("unknown", "--generator", help="Generator label for AI images"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """Evaluate the detector on a directory of real vs AI images."""
    samples = []
    for p in sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png")):
        samples.append({"image": load_image(p), "label": 0, "generator": "real-camera", "path": str(p)})
    for p in sorted(ai_dir.glob("*.jpg")) + sorted(ai_dir.glob("*.png")):
        samples.append({"image": load_image(p), "label": 1, "generator": generator, "path": str(p)})
    if not samples:
        console.print("[red]No images found.[/red]")
        raise typer.Exit(1)
    pipeline = DetectorPipeline()
    report = evaluate(pipeline, samples)
    if json_out:
        console.print_json(json.dumps(report.to_dict()))
        return
    table = Table(title="SAI evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("AUROC", f"{report.auroc:.4f}")
    table.add_row("Accuracy", f"{report.accuracy:.4f}")
    table.add_row("Refusal-aware accuracy", f"{report.refusal_aware_accuracy:.4f}")
    table.add_row("Refusal rate", f"{report.refusal_rate:.4f}")
    table.add_row("ECE", f"{report.ece:.4f}")
    console.print(table)
    if report.per_generator_auroc:
        gen_table = Table(title="Per-generator AUROC")
        gen_table.add_column("Generator", style="cyan")
        gen_table.add_column("AUROC", style="magenta")
        for g, a in report.per_generator_auroc.items():
            gen_table.add_row(g, f"{a:.4f}")
        console.print(gen_table)


@app.command()
def calibrate(
    real_dir: Path = typer.Argument(..., exists=True),
    ai_dir: Path = typer.Argument(..., exists=True),
    out: Path = typer.Option(Path("calibration.json"), "--out", help="Where to save calibration"),
) -> None:
    """Fit temperature scaling on a labeled set and save it."""
    samples = []
    for p in sorted(real_dir.glob("*.jpg")) + sorted(real_dir.glob("*.png")):
        samples.append({"image": load_image(p), "label": 0})
    for p in sorted(ai_dir.glob("*.jpg")) + sorted(ai_dir.glob("*.png")):
        samples.append({"image": load_image(p), "label": 1})
    if not samples:
        console.print("[red]No images found.[/red]")
        raise typer.Exit(1)
    pipeline = DetectorPipeline()
    raw_scores = []
    labels = []
    for s in samples:
        r = pipeline.detect(s["image"])
        raw_scores.append(r.raw_score)
        labels.append(s["label"])
    T = pipeline.fit_calibration(raw_scores, labels)
    out.write_text(json.dumps({"temperature": T}))
    console.print(f"[green]Fitted temperature: {T:.4f}[/green]")
    console.print(f"[green]Saved to {out}[/green]")


if __name__ == "__main__":
    app()
