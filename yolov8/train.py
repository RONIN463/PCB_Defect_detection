from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "yolov8_pcb_dataset" / "pcb.yaml"
RUNS_DIR = ROOT / "runs"


def main() -> None:
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=200,
        imgsz=640,
        batch=32,
        device=0,
        patience=20,
        project=str(RUNS_DIR),
        name="yolov8n_pcb",
        optimizer="AdamW",
        seed=42,
        cos_lr=True,
        plots=True,
        workers=4,
    )

    print("训练完成！")


if __name__ == "__main__":
    main()
