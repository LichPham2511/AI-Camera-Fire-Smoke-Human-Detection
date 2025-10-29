from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

from ultralytics import YOLO

#   DEFAULT_WEIGHTS = "AUTO"                          # pick newest *.pt in this folder
DEFAULT_WEIGHTS: Union[str, Path] = r"d:\\AI-YOLO\\best.pt"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run a trained YOLO model on webcam/video/image.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--weights",
		type=str,
		default=None,
		help="Override weights path (.pt). If omitted, uses DEFAULT_WEIGHTS in this script.",
	)
	parser.add_argument(
		"--source",
		type=str,
		default="0",
		help="Inference source: 0 for default webcam, or a video/image path/URL",
	)
	parser.add_argument(
		"--conf",
		type=float,
		default=0.25,
		help="Confidence threshold",
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=640,
		help="Inference image size",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="",
		help="Device to run on: '' (auto), 'cpu', or 'cuda:0'",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Show result window(s)",
	)
	parser.add_argument(
		"--save",
		action="store_true",
		help="Save annotated results to runs directory",
	)
	parser.add_argument(
		"--check",
		action="store_true",
		help="Only check that the weights file can be resolved, then exit",
	)
	return parser.parse_args()


def resolve_weights_path(weights_spec: Union[str, Path]) -> Path:
	"""Resolve a weights path.

	Tries the following in order:
	- If absolute path: use it directly
	- As given, relative to CWD
	- Relative to this script's directory
	- Special value 'AUTO': newest .pt in this script's directory
	"""
	spec = Path(str(weights_spec))
	# Special AUTO handling first
	if str(weights_spec).upper() == "AUTO":
		candidates = sorted(
			Path(__file__).parent.glob("*.pt"),
			key=lambda p: p.stat().st_mtime,
			reverse=True,
		)
		if candidates:
			return candidates[0]
		raise FileNotFoundError("AUTO could not find any .pt files next to the script")

	# Build candidate list
	candidates = []
	if spec.is_absolute():
		candidates.append(spec)
	else:
		candidates.append(spec)
		candidates.append(Path(__file__).parent / spec)

	for c in candidates:
		if c.is_file():
			return c

	raised = ", ".join(str(c) for c in candidates)
	raise FileNotFoundError(
		f"Could not find weights file for spec '{weights_spec}'. Tried: {raised}"
	)


def coerce_source(src: str) -> Union[int, str]:
	"""Try to coerce a source string to int if it's a camera index like '0'."""
	try:
		return int(src)
	except ValueError:
		return src


def main() -> None:
	args = parse_args()

	# Choose weights: CLI override or hard-coded default
	weights_spec: Union[str, Path] = args.weights if args.weights else DEFAULT_WEIGHTS
	try:
		weights_path = resolve_weights_path(weights_spec)
	except FileNotFoundError as e:
		print(str(e))
		print("Hint: Set DEFAULT_WEIGHTS at the top of yolo.py to your .pt file,\n"
		      "      or pass --weights <path-to-weights.pt> when running.")
		sys.exit(1)

	print(f"Using weights: {weights_path}")
	if args.check:
		print("Weights file resolved successfully. Exiting due to --check.")
		return

	# Load model
	model = YOLO(str(weights_path))

	source = coerce_source(args.source)
	print(f"Starting inference on source: {source}")

	# Use Ultralytics built-in inference loop; it handles webcam/video/image
	# Press 'q' in the window to quit when --show is used.
	results = model.predict(
		source=source,
		conf=args.conf,
		imgsz=args.imgsz,
		device=args.device if args.device else None,
		show=args.show,
		save=args.save,
		stream=False,
		verbose=True,
	)

	# If not streaming, results is a list. Print a short summary.
	if isinstance(results, list) and results:
		names = model.names if hasattr(model, "names") else None
		first = results[0]
		print("Done. Example detections (first frame):")
		try:
			boxes = first.boxes
			if boxes is not None and len(boxes) > 0:
				for i, b in enumerate(boxes):
					cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else -1
					conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
					label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
					print(f"  #{i+1}: {label} @ {conf:.2f}")
			else:
				print("  No detections in the first result.")
		except Exception:
			# Be resilient if result structure changes across versions
			print("  Parsed results summary unavailable (non-critical).")


if __name__ == "__main__":
	main()

