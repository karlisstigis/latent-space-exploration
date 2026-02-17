from __future__ import annotations

"""
Activation capture + caching for ComfyUI.

Key features:
- OUTPUT node capture (so it can run without downstream UMAP).
- Disk-streaming capture via NumPy memmap to avoid RAM blowups.
- Manifest-based caching: if the exact same run is already on disk, it returns instantly.
- Loader node: load a previously captured activation directory (with manifest) into the graph.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, List, Optional
from collections import defaultdict
import os
import json
import time
import hashlib
import gc

def _check_interrupt():
    """Cooperative interrupt for ComfyUI.
    Call frequently inside long loops so Comfy can stop the run.
    """
    try:
        import comfy.model_management as mm  # type: ignore
        if hasattr(mm, "throw_exception_if_processing_interrupted"):
            mm.throw_exception_if_processing_interrupted()
        elif hasattr(mm, "processing_interrupted") and mm.processing_interrupted():
            raise RuntimeError("Interrupted")
    except ModuleNotFoundError:
        return


try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

# ComfyUI helper for consistent output paths
try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None  # type: ignore

from .clip_model import ClipModelBundle


REQS = ["torch>=2.1.0", "numpy>=1.23.0"]


@dataclass
class ActivationRun:
    # In-memory tensors (only when store_mode == 'memory')
    cls_per_step: Dict[str, "torch.Tensor"] = field(default_factory=dict)
    capture_points: List[str] = field(default_factory=list)

    # Disk-backed run info (store_mode == 'disk')
    store_mode: str = "disk"  # 'disk' | 'memory'
    output_dir: str = ""
    file_prefix: str = ""
    saved_files: Dict[str, str] = field(default_factory=dict)

    # For caching / reproducibility
    config_hash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


# --------------------------
# Manifest + caching helpers
# --------------------------

def _hash_config(d: Dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _resolve_output_dir(output_dir: str) -> str:
    base_dir = folder_paths.get_output_directory() if folder_paths is not None else "outputs"
    resolved = output_dir or base_dir
    if not os.path.isabs(resolved):
        resolved = os.path.join(base_dir, resolved)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _manifest_path(out_dir: str, prefix: str) -> str:
    pfx = f"{prefix}_" if prefix else ""
    return os.path.join(out_dir, f"{pfx}manifest.json")


def _load_manifest(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_manifest(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _validate_saved_files(saved_files: Dict[str, str], n_total: int) -> bool:
    if np is None:
        return False
    if not isinstance(saved_files, dict) or not saved_files:
        return False
    for step_name, p in saved_files.items():
        if not p or not os.path.exists(p):
            return False
        try:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 2:
                return False
            if int(arr.shape[0]) != int(n_total):
                return False
        except Exception:
            return False
    return True


# --------------------------
# Disk streaming writer
# --------------------------

class _MemmapWriter:
    def __init__(self, out_dir: str, prefix: str, dtype: str = "float16", overwrite: bool = False):
        if np is None:
            raise RuntimeError("NumPy is required for disk streaming mode.")
        self.out_dir = out_dir
        self.prefix = f"{prefix}_" if prefix else ""
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.overwrite = bool(overwrite)
        self._mmaps: Dict[str, Any] = {}
        self._files: Dict[str, Any] = {}
        self._offset = 0
        self._n_total: Optional[int] = None

    def set_total(self, n_total: int) -> None:
        self._n_total = int(n_total)

    def write_batch(self, step_name: str, batch_arr: "np.ndarray") -> None:
        if self._n_total is None:
            raise RuntimeError("MemmapWriter total size not set.")
        if step_name not in self._mmaps:
            path = os.path.join(self.out_dir, f"{self.prefix}{step_name}_promptsXneurons.npy")
            if getattr(self, 'overwrite', False) and os.path.exists(path):
                # Windows memmap overwrite requires deleting the old file first
                try:
                    f_old = self._files.get(step_name)
                    if f_old is not None:
                        try:
                            f_old.close()
                        except Exception:
                            pass
                    self._files.pop(step_name, None)
                except Exception:
                    pass
                try:
                    os.remove(path)
                except Exception as e:
                    raise RuntimeError(f"Could not overwrite existing file: {path}") from e
            mm = np.lib.format.open_memmap(
                path, mode="w+", dtype=self.dtype, shape=(self._n_total, int(batch_arr.shape[1]))
            )
            self._mmaps[step_name] = mm
            # Keep an OS-level file handle for reliable fsync on Windows
            try:
                self._files[step_name] = open(path, "r+b", buffering=0)
            except Exception:
                self._files[step_name] = None

        mm = self._mmaps[step_name]
        b = int(batch_arr.shape[0])
        mm[self._offset : self._offset + b, :] = batch_arr.astype(self.dtype, copy=False)

    def advance(self, batch_size: int) -> None:
        self._offset += int(batch_size)

    def flush_all(self) -> None:
        """Flush memmaps and force OS writeback.
        On Windows, relying on mm.flush() alone can leave most data in RAM.
        We therefore fsync the underlying file handles we opened for each memmap.
        """
        for name, mm in self._mmaps.items():
            try:
                mm.flush()
            except Exception:
                pass
            f = self._files.get(name)
            if f is not None:
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass

    def finalize(self) -> Dict[str, str]:
        saved: Dict[str, str] = {}
        # Close file handles (and flush/fsync one last time)
        for name, f in list(self._files.items()):
            if f is None:
                continue
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

        for step_name, mm in self._mmaps.items():
            try:
                mm.flush()
            except Exception:
                pass
            saved[step_name] = getattr(mm, "filename", "")
        return saved


# --------------------------
# Hook registration
# --------------------------

def _register_text_internal_hooks(
    model: "torch.nn.Module",
    activations: DefaultDict[str, List["torch.Tensor"]],
    latest_token_pos_getter: Callable[[], "torch.Tensor"],
    *,
    writer: Optional[_MemmapWriter] = None,
    capture_profile: str = "full",  # 'minimal' | 'standard' | 'full'
) -> List[Any]:
    handles: List[Any] = []

    profile = (capture_profile or "full").lower().strip()
    if profile == "minimal":
        capture_points = ["resid_pre", "out"]
    elif profile == "standard":
        capture_points = ["resid_pre", "resid_mid", "out"]
    else:
        capture_points = [
            "resid_pre", "resid_mid", "out",
            "ln1", "attn", "ln2",
            "mlp_fc1", "mlp_act", "mlp_fc2",
        ]

    def make_hook(name: str, use_input: bool = False):
        def hook(module, input, output=None):
            out = input[0] if use_input else output
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out is None or out.dim() != 3:
                return

            idx = latest_token_pos_getter().to(out.device).long()
            if idx.numel() == 0:
                return
            B = int(idx.shape[0])

            # Normalize layout to [B, T, D] (OpenCLIP sometimes uses [T, B, D])
            if out.shape[0] == B:
                x = out
            elif out.shape[1] == B:
                x = out.permute(1, 0, 2)
            else:
                return

            imax = int(idx.max())
            if imax >= x.shape[1]:
                return

            rows = torch.arange(B, device=x.device)
            vec = x[rows, idx, :]  # [B, D]

            if writer is None:
                activations[name].append(vec.detach().cpu())
            else:
                vec_cpu = vec.detach().to("cpu")
                writer.write_batch(name, vec_cpu.numpy())
        return hook

    # OpenCLIP text tower: transformer.resblocks
    for i, block in enumerate(model.transformer.resblocks):
        base = f"text_block_{i:02d}"

        if "resid_pre" in capture_points:
            handles.append(block.ln_1.register_forward_pre_hook(
                make_hook(base + ".resid_pre", use_input=True)
            ))
        if "resid_mid" in capture_points:
            handles.append(block.ln_2.register_forward_pre_hook(
                make_hook(base + ".resid_mid", use_input=True)
            ))
        if "out" in capture_points:
            handles.append(block.register_forward_hook(make_hook(base + ".out")))

        if "ln1" in capture_points:
            handles.append(block.ln_1.register_forward_hook(make_hook(base + ".ln1")))
        if "attn" in capture_points:
            handles.append(block.attn.register_forward_hook(make_hook(base + ".attn")))
        if "ln2" in capture_points:
            handles.append(block.ln_2.register_forward_hook(make_hook(base + ".ln2")))

        # MLP is usually Sequential
        if hasattr(block, "mlp") and isinstance(block.mlp, torch.nn.Sequential):
            if "mlp_fc1" in capture_points and len(block.mlp) > 0:
                handles.append(block.mlp[0].register_forward_hook(make_hook(base + ".mlp_fc1")))
            if "mlp_act" in capture_points and len(block.mlp) > 1:
                handles.append(block.mlp[1].register_forward_hook(make_hook(base + ".mlp_act")))
            if "mlp_fc2" in capture_points and len(block.mlp) > 2:
                handles.append(block.mlp[2].register_forward_hook(make_hook(base + ".mlp_fc2")))

    return handles


# --------------------------
# Core capture function
# --------------------------

def collect_clip_text_activations(
    bundle: ClipModelBundle,
    texts: List[str],
    batch_size: int = 64,
    *,
    store_mode: str = "disk",          # 'disk' or 'memory'
    output_dir: str = "raw_activations",
    file_prefix: str = "",
    dtype: str = "float16",            # disk mode only
    capture_profile: str = "full",
    overwrite: bool = False,
    gc_interval: int = 200,
    flush_interval: int = 500,
    progress_interval: int = 200,
) -> ActivationRun:
    if torch is None:
        raise RuntimeError("Torch is required.")

    store_mode = (store_mode or "disk").lower().strip()
    device = torch.device(bundle.device)
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.eval()

    n_total = len(texts)
    resolved_dir = _resolve_output_dir(output_dir) if store_mode == "disk" else ""
    manifest_p = _manifest_path(resolved_dir, file_prefix) if store_mode == "disk" else ""

    config: Dict[str, Any] = {
        "version": 2,
        "store_mode": store_mode,
        "dtype": dtype,
        "capture_profile": capture_profile,
        "batch_size": int(batch_size),
        "n_total": int(n_total),
        # best-effort identifiers
        "clip_model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "file_prefix": file_prefix,
    }
    config_hash = _hash_config(config)
    config["config_hash"] = config_hash

    # Cache hit: return immediately if everything is already on disk
    if store_mode == "disk" and overwrite:
        # Remove old manifest/progress so the run is clearly fresh
        try:
            if os.path.exists(manifest_p):
                os.remove(manifest_p)
        except Exception:
            pass
        try:
            ppath = os.path.join(resolved_dir, f"{file_prefix + '_' if file_prefix else ''}progress.json")
            if os.path.exists(ppath):
                os.remove(ppath)
        except Exception:
            pass

    if store_mode == "disk" and not overwrite:
        m = _load_manifest(manifest_p)
        if m and m.get("config_hash") == config_hash:
            saved = m.get("saved_files", {})
            if isinstance(saved, dict) and _validate_saved_files(saved, n_total):
                return ActivationRun(
                    cls_per_step={},
                    capture_points=sorted(saved.keys()),
                    store_mode="disk",
                    output_dir=resolved_dir,
                    file_prefix=file_prefix,
                    saved_files=saved,
                    config_hash=config_hash,
                    config=m,
                )

    activations: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
    latest_token_pos = torch.tensor(0, device=device)

    def latest():
        return latest_token_pos

    writer: Optional[_MemmapWriter] = None
    if store_mode == "disk":
        try:
            writer = _MemmapWriter(resolved_dir, file_prefix, dtype=dtype, overwrite=overwrite)
        except TypeError:
            writer = _MemmapWriter(resolved_dir, file_prefix, dtype=dtype)
            try:
                setattr(writer, 'overwrite', bool(overwrite))
            except Exception:
                pass
        writer.set_total(n_total)

    handles = _register_text_internal_hooks(
        model,
        activations,
        latest,
        writer=writer,
        capture_profile=capture_profile,
    )

    eot_id = tokenizer.eot_token_id

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(batch_texts)

            latest_token_pos = (tokens == eot_id).int().argmax(dim=-1).to(device)

            tokens = tokens.to(device)
            _ = model.encode_text(tokens)

            if writer is not None:
                writer.advance(len(batch_texts))

        # End of CSV chunk
        try:
            del chunk_texts
        except Exception:
            pass
        if gc_interval:
            gc.collect()

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    cls_per_step: Dict[str, torch.Tensor] = {}
    saved_files: Dict[str, str] = {}

    if writer is None:
        for name, ts in activations.items():
            cls_per_step[name] = torch.cat(ts, dim=0)
        capture_points = sorted(cls_per_step.keys())
        return ActivationRun(
            cls_per_step=cls_per_step,
            capture_points=capture_points,
            store_mode="memory",
            output_dir="",
            file_prefix=file_prefix,
            saved_files={},
            config_hash=config_hash,
            config=config,
        )

    # Disk mode
    saved_files = writer.finalize()
    capture_points = sorted(saved_files.keys())

    manifest = {
        **config,
        "created_at": int(time.time()),
        "saved_files": saved_files,
    }
    _write_manifest(manifest_p, manifest)

    return ActivationRun(
        cls_per_step={},
        capture_points=capture_points,
        store_mode="disk",
        output_dir=resolved_dir,
        file_prefix=file_prefix,
        saved_files=saved_files,
        config_hash=config_hash,
        config=manifest,
    )



# --------------------------
# CSV streaming (avoid 1M strings in RAM)
# --------------------------

def _iter_csv_text_batches(
    csv_path: str,
    text_column: str,
    start: int,
    end: int,
    chunksize: int,
):
    """
    Yields lists of texts from CSV without loading the full file.

    start: inclusive
    end: exclusive (-1 = until EOF)
    """
    if np is None:
        raise RuntimeError("NumPy required.")
    import pandas as pd

    start = max(0, int(start))
    end = int(end)

    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=[text_column], chunksize=int(chunksize)):
        if text_column not in chunk.columns:
            raise ValueError(f"Missing required column '{text_column}' in {csv_path}")

        chunk_len = len(chunk)
        chunk_start = seen
        chunk_end = seen + chunk_len

        desired_end = end if end != -1 else float("inf")

        if chunk_end <= start:
            seen += chunk_len
            continue
        if chunk_start >= desired_end:
            break

        take_from = max(0, start - chunk_start)
        take_to = chunk_len if end == -1 else min(chunk_len, end - chunk_start)

        sub = chunk.iloc[take_from:take_to]
        texts = sub[text_column].astype(str).tolist()
        if texts:
            yield texts

        seen += chunk_len


def collect_clip_text_activations_from_csv(
    bundle: ClipModelBundle,
    csv_path: str,
    text_column: str = "text",
    start: int = 0,
    end: int = -1,
    resume_from: int = 0,
    chunksize: int = 200_000,
    batch_size: int = 64,
    *,
    store_mode: str = "disk",
    output_dir: str = "raw_activations",
    file_prefix: str = "",
    dtype: str = "float16",
    capture_profile: str = "full",
    overwrite: bool = False,
    gc_interval: int = 200,
    flush_interval: int = 500,
    progress_interval: int = 200,
) -> ActivationRun:
    """
    Same as collect_clip_text_activations, but streams prompts directly from a CSV to avoid
    holding a million Python strings in RAM.
    """
    # First pass: count how many rows we will process (needed for memmap prealloc)
    import pandas as pd

    start_i = max(0, int(start))
    end_i = int(end)
    n_total = 0
    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=[text_column], chunksize=int(chunksize)):
        chunk_len = len(chunk)
        chunk_start = seen
        chunk_end = seen + chunk_len
        desired_end = end_i if end_i != -1 else float("inf")

        if chunk_end <= start_i:
            seen += chunk_len
            continue
        if chunk_start >= desired_end:
            break

        take_from = max(0, start_i - chunk_start)
        take_to = chunk_len if end_i == -1 else min(chunk_len, end_i - chunk_start)
        n_total += max(0, take_to - take_from)
        seen += chunk_len

    resume_from = max(0, int(resume_from))
    if resume_from:
        n_total = max(0, n_total - resume_from)

    if n_total <= 0:
        raise RuntimeError("No rows selected from CSV (after resume_from). Check start/end/resume_from.")

    # Now do capture but iterate CSV chunks -> tokenize in batches
    if torch is None:
        raise RuntimeError("Torch is required.")

    store_mode = (store_mode or "disk").lower().strip()
    device = torch.device(bundle.device)
    tokenizer = bundle.tokenizer
    model = bundle.model
    model.eval()

    resolved_dir = _resolve_output_dir(output_dir) if store_mode == "disk" else ""
    manifest_p = _manifest_path(resolved_dir, file_prefix) if store_mode == "disk" else ""

    config: Dict[str, Any] = {
        "version": 3,
        "store_mode": store_mode,
        "dtype": dtype,
        "capture_profile": capture_profile,
        "batch_size": int(batch_size),
        "n_total": int(n_total),
        "csv_path": os.path.abspath(csv_path),
        "text_column": text_column,
        "start": int(start),
        "end": int(end),
        "resume_from": int(resume_from),
        "chunksize": int(chunksize),
        "clip_model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "file_prefix": file_prefix,
    }
    config_hash = _hash_config(config)
    config["config_hash"] = config_hash

    if store_mode == "disk" and overwrite:
        # Remove old manifest/progress so the run is clearly fresh
        try:
            if os.path.exists(manifest_p):
                os.remove(manifest_p)
        except Exception:
            pass
        try:
            ppath = os.path.join(resolved_dir, f"{file_prefix + '_' if file_prefix else ''}progress.json")
            if os.path.exists(ppath):
                os.remove(ppath)
        except Exception:
            pass

    if store_mode == "disk" and not overwrite:
        m = _load_manifest(manifest_p)
        if m and m.get("config_hash") == config_hash:
            saved = m.get("saved_files", {})
            if isinstance(saved, dict) and _validate_saved_files(saved, n_total):
                return ActivationRun(
                    cls_per_step={},
                    capture_points=sorted(saved.keys()),
                    store_mode="disk",
                    output_dir=resolved_dir,
                    file_prefix=file_prefix,
                    saved_files=saved,
                    config_hash=config_hash,
                    config=m,
                )

    activations: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
    latest_token_pos = torch.tensor(0, device=device)

    def latest():
        return latest_token_pos

    writer: Optional[_MemmapWriter] = None
    if store_mode == "disk":
        try:
            writer = _MemmapWriter(resolved_dir, file_prefix, dtype=dtype, overwrite=overwrite)
        except TypeError:
            writer = _MemmapWriter(resolved_dir, file_prefix, dtype=dtype)
            try:
                setattr(writer, 'overwrite', bool(overwrite))
            except Exception:
                pass
        writer.set_total(n_total)

    handles = _register_text_internal_hooks(
        model, activations, latest, writer=writer, capture_profile=capture_profile
    )

    eot_id = tokenizer.eot_token_id

    wrote_any = False
    processed = 0

    with torch.no_grad():
        # Apply resume offset within the selected range
        effective_start = int(start) + max(0, int(resume_from))
        for chunk_texts in _iter_csv_text_batches(csv_path, text_column, effective_start, end, chunksize):
            # Break chunk_texts into batches
            for s in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[s : s + batch_size]
                tokens = tokenizer(batch_texts)

                latest_token_pos = (tokens == eot_id).int().argmax(dim=-1).to(device)

                tokens = tokens.to(device)
                _ = model.encode_text(tokens)

                if writer is not None:
                    writer.advance(len(batch_texts))
                    wrote_any = True

                processed += len(batch_texts)
                if store_mode == "disk" and progress_interval and (processed // max(1, batch_size)) % progress_interval == 0:
                    try:
                        ppath = os.path.join(resolved_dir, f"{file_prefix + '_' if file_prefix else ''}progress.json")
                        with open(ppath, 'w', encoding='utf-8') as pf:
                            json.dump({"processed": int(processed), "n_total": int(n_total), "effective_start": int(effective_start)}, pf)
                    except Exception:
                        pass

                # Encourage timely release of temporary objects (helps with long runs)
                del tokens
                del batch_texts
                if gc_interval and (processed // max(1, batch_size)) % gc_interval == 0:
                    gc.collect()
                if writer is not None and flush_interval and (processed // max(1, batch_size)) % flush_interval == 0:
                    _check_interrupt()
                    writer.flush_all()

        # End of CSV chunk
        try:
            del chunk_texts
        except Exception:
            pass
        if gc_interval:
            gc.collect()

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    cls_per_step: Dict[str, torch.Tensor] = {}
    saved_files: Dict[str, str] = {}

    if writer is None:
        for name, ts in activations.items():
            cls_per_step[name] = torch.cat(ts, dim=0)
        capture_points = sorted(cls_per_step.keys())
        return ActivationRun(
            cls_per_step=cls_per_step,
            capture_points=capture_points,
            store_mode="memory",
            output_dir="",
            file_prefix=file_prefix,
            saved_files={},
            config_hash=config_hash,
            config=config,
        )

    saved_files = writer.finalize()
    capture_points = sorted(saved_files.keys())

    manifest = {**config, "created_at": int(time.time()), "saved_files": saved_files, "n_written": int(n_total), "complete": True}
    _write_manifest(manifest_p, manifest)

    return ActivationRun(
        cls_per_step={},
        capture_points=capture_points,
        store_mode="disk",
        output_dir=resolved_dir,
        file_prefix=file_prefix,
        saved_files=saved_files,
        config_hash=config_hash,
        config=manifest,
    )
# --------------------------
# ComfyUI nodes
# --------------------------

class ClipActivationRecorder:
    """
    OUTPUT node: capture activations and write them to disk (or memory for small tests).
    Includes caching: will skip work if the same capture already exists.
    """
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_model": ("MODEL", {"tooltip": "OpenCLIP model bundle (output of your CLIP loader node)."}),
                "dataset": ("MODEL", {"tooltip": "PromptDataset from loader. If loader is in stream mode, this node streams prompts from CSV automatically."}),
                "batch_size": ("INT", {"default": 64, "min": 1, "max": 4096,
                                       "tooltip": "How many prompts to run per forward pass (GPU VRAM)."}),
                "store_mode": (["disk", "memory"], {"default": "disk",
                                                    "tooltip": "Disk streams activations to .npy files (recommended). Memory keeps tensors in RAM (small tests only)."}),
                "output_dir": ("STRING", {"default": "clip_activations",
                                          "tooltip": "Directory to write .npy activation files and a manifest.json. Absolute path recommended for large runs."}),
                "file_prefix": ("STRING", {"default": "",
                                           "tooltip": "Prefix added to all saved files (useful for multiple runs in the same folder)."}),
                "dtype": (["float16", "float32"], {"default": "float16",
                                                   "tooltip": "Numeric precision for saved activations. float16 halves disk usage and is usually sufficient."}),
                "capture_profile": (["full", "standard", "minimal"], {"default": "full",
                                                                      "tooltip": "Which internal points to capture per transformer block. Full = most detail + most disk."}),
                "overwrite": ("BOOLEAN", {"default": False,
                                          "tooltip": "If false, will reuse a previous capture if an identical manifest exists. If true, recaptures and overwrites files."}),
                "gc_interval": ("INT", {"default": 200, "min": 0, "max": 100000, "step": 1,
                                        "tooltip": "Run Python garbage collection every N batches to reduce long-run RAM creep. 0 disables."}),
                "flush_interval": ("INT", {"default": 500, "min": 0, "max": 100000, "step": 1,
                                           "tooltip": "Flush memmap buffers to disk every N batches to reduce Windows file-cache RAM growth. 0 disables."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("activations", "output_dir")
    FUNCTION = "run"
    CATEGORY = "CLIP Intermediate Export"

    def run(
        self,
        clip_model: ClipModelBundle,
        dataset: Any,
        batch_size: int,
        store_mode: str,
        output_dir: str,
        file_prefix: str,
        dtype: str,
        capture_profile: str,
        overwrite: bool,
        gc_interval: int,
        flush_interval: int,
    ):
        ds_mode = str(getattr(dataset, "load_mode", "memory")).strip().lower()
        if ds_mode == "stream":
            csv_path = str(getattr(dataset, "csv_path", "") or getattr(dataset, "path", ""))
            text_column = str(getattr(dataset, "text_column", "text"))
            start = int(getattr(dataset, "start", 0))
            end = int(getattr(dataset, "end", -1))
            chunksize = int(getattr(dataset, "chunksize", 200_000))
            if not csv_path:
                raise RuntimeError("Streaming dataset is missing csv_path/path.")
            run = collect_clip_text_activations_from_csv(
                clip_model,
                csv_path=csv_path,
                text_column=text_column,
                start=start,
                end=end,
                chunksize=chunksize,
                batch_size=batch_size,
                store_mode=store_mode,
                output_dir=output_dir,
                file_prefix=file_prefix,
                dtype=dtype,
                capture_profile=capture_profile,
                overwrite=overwrite,
                gc_interval=gc_interval,
                flush_interval=flush_interval,
            )
        else:
            texts = getattr(dataset, "texts", None)
            if texts is None:
                raise RuntimeError(
                    "Dataset has no in-memory texts. Set loader load_mode='memory' or use a stream-capable dataset."
                )
            run = collect_clip_text_activations(
                clip_model,
                texts,
                batch_size=batch_size,
                store_mode=store_mode,
                output_dir=output_dir,
                file_prefix=file_prefix,
                dtype=dtype,
                capture_profile=capture_profile,
                overwrite=overwrite,
                gc_interval=gc_interval,
                flush_interval=flush_interval,
            )
        return (run, run.output_dir or "")



# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ClipActivationRecorder": ClipActivationRecorder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipActivationRecorder": "CIE: Capture Activations",
}
