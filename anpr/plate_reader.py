"""Owner: person 3. Vehicle crop in, characters + per-character confidence out.

Plate localization: fast-alpr's detector (open-image-models).
    https://github.com/ankandrew/open-image-models
    pip install "open-image-models[onnx]"

Character reading: Awiros-ANPR-OCR (PP-OCRv5 / SVTR_HGNet) via PaddleOCR,
with CTC gap-recovery for characters the model's collapse step discards.
    https://huggingface.co/surendran0m07/anpr-ocr
    https://github.com/PaddlePaddle/PaddleOCR

Anything unknown is None. An unreadable slot is '?' with conf 0.0.

Dual-purpose: imported by the pipeline (read_plate), or run directly for
debugging (python plate_reader.py --image_path ... — see --mode below).
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config  # running as a standalone script

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Plate localization — fast-alpr's detector (open-image-models)
# ---------------------------------------------------------------------------
_PLATE_LABEL = "License Plate"
_plate_detector = None


def _get_plate_detector():
    """Loaded on first use, not at import. A top-level import here meant a
    missing OCR package stopped main.py from starting at all, so nobody could
    run the camera, detector, database or linker either."""
    global _plate_detector
    if _plate_detector is None:
        try:
            from open_image_models import create_detector
        except ImportError as exc:
            raise RuntimeError(
                'plate detector not installed. Run:  pip install "open-image-models[onnx]"'
            ) from exc
        _plate_detector = create_detector(
            getattr(config, "PLATE_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end"),
            conf_thresh=getattr(config, "PLATE_DETECTOR_CONF_THRESH", None),
        )
    return _plate_detector


def _localize_plate(crop: np.ndarray):
    """Runs the plate detector on a vehicle crop.
    Returns the best plate box as (x1, y1, x2, y2), or None if nothing found.
    """
    detector = _get_plate_detector()
    detections = detector.predict(crop)
    if not detections:
        return None

    # Safety net in case config swaps in a multi-class detector later.
    plate_detections = [d for d in detections if d.label == _PLATE_LABEL] or detections
    best = max(plate_detections, key=lambda d: d.confidence)

    box = best.bounding_box.clamp(crop.shape[1], crop.shape[0])
    if box.is_empty:
        return None
    return box.xyxy


# ---------------------------------------------------------------------------
# Character reading — Awiros-ANPR-OCR via PaddleOCR, with CTC gap recovery
# ---------------------------------------------------------------------------
_CTC_NUM_CLASSES = 64
_NRTR_NUM_CLASSES = 67  # NRTRHead internally adds +1, so 67 -> 68 to match weights

_MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": _CTC_NUM_CLASSES,
                "NRTRLabelDecode": _NRTR_NUM_CLASSES,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 1e-05},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    },
}
_IMAGE_SHAPE = [3, 48, 320]

_paddle = None
_ocr_model = None
_ocr_post_process = None
_ocr_loaded_with = None  # tracks which (weights, dict, device, paddleocr_dir) is currently loaded


def _find_paddleocr(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates += [_MODULE_DIR / "PaddleOCR", _MODULE_DIR, Path.cwd(), Path.cwd() / "PaddleOCR"]
    for c in candidates:
        if (c / "ppocr" / "__init__.py").is_file():
            return c
    return None


def _ensure_paddleocr(explicit_path=None):
    """Put a PaddleOCR checkout on sys.path, or say how to get one.

    This used to `git clone` PaddleOCR on the first plate read. That is a large
    download in the middle of a run, which is the worst possible moment for it
    on venue wifi. Cloning is now a setup step, done once, on purpose.
    """
    root = _find_paddleocr(explicit_path)
    if root is None:
        raise RuntimeError(
            "PaddleOCR not found. Run this once, from the repo root:"
            "\n    git clone --depth 1 "
            "https://github.com/PaddlePaddle/PaddleOCR.git "
            f"{_MODULE_DIR / 'PaddleOCR'}"
            "\nor set config.PADDLEOCR_DIR to an existing checkout."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _ensure_probs(logits):
    """Softmax only if the model's raw output isn't already normalized."""
    sums = logits.sum(axis=-1)
    if np.allclose(sums, 1.0, atol=1e-2):
        return logits
    return _softmax(logits, axis=-1)


def _preprocess(img_bgr, target_shape):
    """Plate crop -> the CHW float tensor the recogniser expects.

    Order matters. PaddleOCR normalises first and then pads the remaining
    width with 0.0, which after its own (x/255 - 0.5) / 0.5 corresponds to
    mid grey. Padding with black pixels first and normalising afterwards puts
    -1.0 there instead, which is not what these weights were trained on, and
    almost every plate is narrower than the 320px target so almost every read
    was affected. Keep this matching ppocr/data/imaug/rec_img_aug.py.
    """
    channels, target_h, target_w = target_shape
    img_h, img_w = img_bgr.shape[:2]

    # max(1, ...): a very tall, narrow box rounds to zero width and cv2.resize
    # raises, which used to end the whole run.
    new_w = max(1, min(int(img_w * (target_h / img_h)), target_w))
    resized = cv2.resize(img_bgr, (new_w, target_h))

    norm = resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
    norm = (norm - 0.5) / 0.5

    padded = np.zeros((channels, target_h, target_w), dtype=np.float32)
    padded[:, :, :new_w] = norm
    return padded


def _load_awiros_model(weights_path=None, dict_path=None, device=None, paddleocr_dir=None, force=False):
    """Lazily loads paddle + the Awiros weights once per process.
    Pipeline callers use config.py's settings; the CLI can override them.
    """
    global _paddle, _ocr_model, _ocr_post_process, _ocr_loaded_with

    weights_path = str(weights_path or getattr(config, "AWIROS_WEIGHTS_PATH", _MODULE_DIR / "model.safetensors"))
    dict_path = str(dict_path or getattr(config, "AWIROS_DICT_PATH", _MODULE_DIR / "en_dict.txt"))
    device = device or getattr(config, "OCR_DEVICE", "gpu")
    paddleocr_dir = paddleocr_dir or getattr(config, "PADDLEOCR_DIR", None)

    key = (weights_path, dict_path, device, paddleocr_dir)
    if _ocr_model is not None and _ocr_loaded_with == key and not force:
        return

    _ensure_paddleocr(paddleocr_dir)

    import paddle
    from ppocr.modeling.architectures import build_model as ppocr_build_model
    from ppocr.postprocess import build_post_process
    from safetensors.numpy import load_file

    if device == "gpu" and not paddle.is_compiled_with_cuda():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    paddle.set_device(device)

    post_process = build_post_process({
        "name": "CTCLabelDecode",
        "character_dict_path": dict_path,
        "use_space_char": True,
    })

    model_config = copy.deepcopy(_MODEL_CONFIG)
    model = ppocr_build_model(model_config["Architecture"])
    model.eval()

    np_state = load_file(weights_path)
    model.set_state_dict({k: paddle.to_tensor(v) for k, v in np_state.items()})

    _paddle = paddle
    _ocr_model = model
    _ocr_post_process = post_process
    _ocr_loaded_with = key
    print(f"Loaded Awiros weights from {weights_path}")


def _ctc_decode_with_gaps(probs, character_list, blank_idx=0,
                           gap_prob_threshold=0.15, min_char_confidence=0.30,
                           gap_width_multiplier=1.8, recover=True, scan_gaps=True):
    """Greedy CTC decode, optionally with gap-based character recovery.

    recover=False: plain CTC greedy decode, no downgrade, no gap-scanning.
    recover=True: also downgrades any kept char below min_char_confidence
    to ('?', 0.0) — independent of scan_gaps, safe for any plate shape.
    scan_gaps=True: also treats an unusually wide gap between two confirmed
    characters as a likely dropped character and tries to recover it.

    IMPORTANT: a two-line plate has one structurally wide gap (the row
    transition) that this heuristic can't tell apart from a real drop.
    Pass scan_gaps=False for dual-row plates (see _read_awiros).

    Limitation: a drop at the very first/last position has no neighboring
    gap to check against and won't be flagged even with scan_gaps=True.

    Returns (chars, confs, debug); debug tags are
    "kept" / "low_conf" / "recovered" / "unresolved".
    """
    T = probs.shape[0]
    argmax_idx = probs.argmax(axis=1)
    argmax_prob = probs.max(axis=1)

    keep = np.ones(T, dtype=bool)
    keep[1:] = argmax_idx[1:] != argmax_idx[:-1]
    keep &= argmax_idx != blank_idx

    kept_timesteps = np.nonzero(keep)[0]
    if len(kept_timesteps) == 0:
        return [], [], []

    kept_chars = [character_list[i] for i in argmax_idx[kept_timesteps]]
    kept_confs = argmax_prob[kept_timesteps].tolist()

    if not recover:
        return kept_chars, [float(c) for c in kept_confs], ["kept"] * len(kept_chars)

    spacings = np.diff(kept_timesteps)
    typical_gap = np.median(spacings) if len(spacings) > 0 else 1.0

    chars, confs, debug = [], [], []
    for i, t in enumerate(kept_timesteps):
        char, conf, tag = kept_chars[i], float(kept_confs[i]), "kept"
        if conf < min_char_confidence:
            char, conf, tag = "?", 0.0, "low_conf"
        chars.append(char)
        confs.append(conf)
        debug.append(tag)

        if i == len(kept_timesteps) - 1:
            continue

        next_t = kept_timesteps[i + 1]
        if scan_gaps and (next_t - t) > max(2, typical_gap * gap_width_multiplier):
            best_class, best_prob = None, 0.0
            for tt in range(t + 1, next_t):
                p = probs[tt].copy()
                p[blank_idx] = 0.0
                c = int(p.argmax())
                if p[c] > best_prob:
                    best_class, best_prob = c, float(p[c])

            if best_prob >= gap_prob_threshold:
                # A recovered character is the model's runner-up inside a gap,
                # not something it actually read. Cap its confidence so it can
                # never be the thing that locks a plate; it still contributes
                # to the vote, it just cannot carry the decision alone.
                capped = min(best_prob, getattr(config, "OCR_RECOVERED_MAX_CONF", 0.45))
                chars.append(character_list[best_class])
                confs.append(capped)
                debug.append("recovered")
            else:
                # Don't silently shorten the plate string.
                chars.append("?")
                confs.append(0.0)
                debug.append("unresolved")

    return chars, confs, debug


def _read_awiros(plate_crop: np.ndarray, gap_prob_threshold=None, min_char_confidence=None,
                  recover=True, scan_gaps=None):
    """Runs the Awiros model on an already-localized plate crop.
    Returns (chars, confs, debug), or ([], [], []) if nothing decodable.

    scan_gaps=None (default) auto-decides from the crop's aspect ratio:
    below config.DUAL_ROW_ASPECT_RATIO_THRESHOLD is treated as dual-row
    and gap-scanning is disabled (see _ctc_decode_with_gaps). Pass
    True/False to override.
    """
    _load_awiros_model()

    if scan_gaps is None:
        h, w = plate_crop.shape[:2]
        aspect_ratio = (w / h) if h > 0 else 0.0
        dual_row_threshold = getattr(config, "DUAL_ROW_ASPECT_RATIO_THRESHOLD", 2.0)
        scan_gaps = aspect_ratio >= dual_row_threshold
        gap_width_multiplier = getattr(config, "OCR_GAP_WIDTH_MULTIPLIER", 1.8)
    else:
        gap_width_multiplier = getattr(config, "OCR_GAP_WIDTH_MULTIPLIER", 1.8)

    tensor = _paddle.to_tensor(
        np.expand_dims(_preprocess(plate_crop, _IMAGE_SHAPE), axis=0)
    )
    with _paddle.no_grad():
        preds = _ocr_model(tensor)

    if isinstance(preds, dict):
        pred_tensor = preds.get("ctc", next(iter(preds.values())))
    elif isinstance(preds, (list, tuple)):
        pred_tensor = preds[0]
    else:
        pred_tensor = preds

    raw_logits = pred_tensor.numpy()
    probs = _ensure_probs(raw_logits)[0]

    ignored = (
        _ocr_post_process.get_ignored_tokens()
        if hasattr(_ocr_post_process, "get_ignored_tokens")
        else [0]
    )
    blank_idx = ignored[0] if ignored else 0

    return _ctc_decode_with_gaps(
        probs,
        _ocr_post_process.character,
        blank_idx=blank_idx,
        gap_prob_threshold=gap_prob_threshold if gap_prob_threshold is not None
            else getattr(config, "OCR_GAP_PROB_THRESHOLD", 0.15),
        min_char_confidence=min_char_confidence if min_char_confidence is not None
            else getattr(config, "OCR_MIN_CHAR_CONFIDENCE", 0.30),
        gap_width_multiplier=gap_width_multiplier,
        recover=recover,
        scan_gaps=scan_gaps,
    )


# ---------------------------------------------------------------------------
# Frozen public interface
# ---------------------------------------------------------------------------
_fail_crops_saved = 0


def _save_fail_crop(crop: np.ndarray, reason: str) -> None:
    """Keep a crop we could not read, for the fine-tuning set and risk slide."""
    global _fail_crops_saved
    if not getattr(config, "SAVE_FAIL_CROPS", False):
        return
    if _fail_crops_saved >= getattr(config, "MAX_FAIL_CROPS", 200):
        return
    if crop is None or crop.size == 0:
        return
    try:
        config.FAIL_CROP_DIR.mkdir(parents=True, exist_ok=True)
        path = config.FAIL_CROP_DIR / f"{_fail_crops_saved:04d}_{reason}.jpg"
        cv2.imwrite(str(path), crop)
        _fail_crops_saved += 1
    except Exception as exc:                     # never let bookkeeping break a run
        logger.debug("could not save failing crop: %s", exc)


def _sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_gate(crop: np.ndarray) -> bool:
    """Cheap pre-check before running OCR: min crop size and min sharpness.

    Both halves matter. Each vehicle only gets MAX_OCR_PER_TRACK attempts, and
    a blurry frame spends one of them to produce nothing. The size test is
    against MIN_VEHICLE_PX, not MIN_PLATE_PX -- this sees the whole vehicle,
    and the plate's own size is checked after the detector locates it.
    """
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    if min(h, w) < config.MIN_VEHICLE_PX:
        return False

    if _sharpness(crop) < config.MIN_SHARPNESS:
        _save_fail_crop(crop, "blurry")
        return False

    return True


def read_plate(crop: np.ndarray) -> dict | None:
    """Returns
        {"chars": ['G','J','0','1','A','B','1','2','3','4'],
         "confs": [0.98, 0.97, ...],        # same length as chars
         "plate_box": (x1,y1,x2,y2)}        # relative to the crop
       None if no plate is found or the crop fails the quality gate.
    """
    if not passes_gate(crop):
        return None

    # The contract is "returns None when there is no plate". An exception is
    # not part of that contract, and read_plate runs once per frame per track,
    # so anything thrown here used to end the run for all three videos.
    try:
        plate_box = _localize_plate(crop)
        if plate_box is None:
            _save_fail_crop(crop, "no_plate_found")
            return None

        x1, y1, x2, y2 = plate_box
        plate_crop = crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None

        # Now that we know where the plate is, check whether it is big enough
        # to be worth reading. This is what MIN_PLATE_PX was always meant for.
        if (x2 - x1) < config.MIN_PLATE_PX:
            _save_fail_crop(plate_crop, "plate_too_small")
            return None

        chars, confs, debug = _read_awiros(plate_crop)
        if not chars:
            _save_fail_crop(plate_crop, "no_characters")
            return None

        if all(c == "?" for c in chars):
            _save_fail_crop(plate_crop, "all_unknown")
            return None

    except Exception as exc:
        logger.warning("read_plate failed on one crop, skipping it: %s", exc)
        return None

    return {
        "chars": chars,
        "confs": confs,
        "plate_box": plate_box,
        # Per-character provenance: "kept" / "low_conf" / "recovered" /
        # "unresolved". A "recovered" character is a guess, not something the
        # model read. Voting and the deck both want to know the difference.
        "debug": debug,
    }


# ---------------------------------------------------------------------------
# Standalone CLI — debugging only, never imported/called by the pipeline
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_images(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Path not found: {path}")


def _parse_args():
    p = argparse.ArgumentParser("plate_reader debug CLI")
    p.add_argument("--image_path", required=True,
                    help="Path to a single image or a directory of images.")
    p.add_argument("--mode", choices=["vehicle", "plate"], default="vehicle",
                    help="'vehicle': full detect+read, like read_plate(). "
                         "'plate': skip detection, feed image straight to OCR.")
    p.add_argument("--no_recover", action="store_true",
                    help="Plain CTC decode only, no gap-recovery or low-conf downgrade.")
    p.add_argument("--gap_prob_threshold", type=float, default=None,
                    help="Override config.OCR_GAP_PROB_THRESHOLD for this run.")
    p.add_argument("--min_char_confidence", type=float, default=None,
                    help="Override config.OCR_MIN_CHAR_CONFIDENCE for this run.")
    p.add_argument("--scan_gaps", choices=["auto", "on", "off"], default="auto",
                    help="'auto': decide from aspect ratio. 'on'/'off': force it.")
    p.add_argument("--device", default=None, choices=["gpu", "cpu"],
                    help="Override config.OCR_DEVICE for this run.")
    p.add_argument("--weights", default=None, help="Override model.safetensors path.")
    p.add_argument("--dict_path", default=None, help="Override en_dict.txt path.")
    p.add_argument("--paddleocr_dir", default=None, help="Override PaddleOCR repo path.")
    p.add_argument("--output_json", default="", help="Optional output JSON path.")
    return p.parse_args()


def _main():
    args = _parse_args()

    _load_awiros_model(
        weights_path=args.weights,
        dict_path=args.dict_path,
        device=args.device,
        paddleocr_dir=args.paddleocr_dir,
        force=True,
    )

    image_paths = _collect_images(args.image_path)
    print(f"Found {len(image_paths)} image(s), mode={args.mode}\n")

    results = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"WARNING: Could not read {img_path}, skipping.")
            continue

        plate_box = None
        if args.mode == "vehicle":
            plate_box = _localize_plate(img_bgr)
            if plate_box is None:
                print(f"  {img_path.name}: no plate detected")
                results.append({"image": img_path.name, "chars": [], "confs": [], "plate_box": None})
                continue
            x1, y1, x2, y2 = plate_box
            plate_crop = img_bgr[y1:y2, x1:x2]
        else:
            plate_crop = img_bgr

        scan_gaps_override = {"auto": None, "on": True, "off": False}[args.scan_gaps]
        chars, confs, debug = _read_awiros(
            plate_crop,
            gap_prob_threshold=args.gap_prob_threshold,
            min_char_confidence=args.min_char_confidence,
            recover=not args.no_recover,
            scan_gaps=scan_gaps_override,
        )
        text = "".join(chars)

        flags = [f"@{i}:{tag}" for i, tag in enumerate(debug) if tag in ("recovered", "unresolved", "low_conf")]
        note = f"  [{', '.join(flags)}]" if flags else ""
        box_note = f"  box={plate_box}" if plate_box else ""
        print(f"  {img_path.name}: {text}{box_note}{note}")

        results.append({
            "image": img_path.name,
            "chars": chars,
            "confs": [round(float(c), 4) for c in confs],
            "plate_box": plate_box,
            "debug": debug,
        })

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    _main()
