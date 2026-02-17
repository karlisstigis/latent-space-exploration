import csv
import os
from typing import Any, Dict, Iterator, List

import numpy as np
import umap
from sklearn.decomposition import PCA

try:
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover
    ProgressBar = None  # type: ignore

try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None  # type: ignore

from .activation_capture import ActivationRun

REQS = ["umap-learn>=0.5.5", "scikit-learn>=1.3.0", "numpy>=1.24.0"]


def to_numpy_cpu(x: Any) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def resolve_cls_per_step(activations: ActivationRun) -> Dict[str, Any]:
    """
    Return per-step activations for UMAP builders.

    - Memory mode: use in-memory tensors directly.
    - Disk mode: lazily memory-map arrays from saved_files.
    """
    cls_per_step = getattr(activations, "cls_per_step", None) or {}
    if cls_per_step:
        return cls_per_step

    saved_files = getattr(activations, "saved_files", None) or {}
    if saved_files:
        loaded: Dict[str, Any] = {}
        for step_name, path in sorted(saved_files.items()):
            loaded[step_name] = np.load(path, mmap_mode="r")
        return loaded

    raise RuntimeError(
        "No activation tensors available. Capture activations first (memory mode), "
        "or provide a valid disk-backed ActivationRun with saved_files."
    )


def run_umap_and_write(
    X: np.ndarray,
    step_names: List[str],
    feat_dim_label: int,
    writer: csv.writer,
    ids: List[Any],
    pca: bool,
    pca_dim: int,
    use_seed: bool,
    seed: int,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    extra_data=None,
    extra_keys=None,
) -> None:
    local_dim = X.shape[1]
    if pca:
        max_components = max(1, min(local_dim, X.shape[0] - 1))
        pca_dim = min(pca_dim, max_components)
        pca_model = PCA(n_components=pca_dim)
        X = pca_model.fit_transform(X)
    if use_seed:
        np.random.seed(seed)
    reducer3 = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed if use_seed else None,
        init="random",
    )
    reducer2 = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed if use_seed else None,
        init="random",
    )
    coords3 = reducer3.fit_transform(X)
    coords2 = reducer2.fit_transform(X)
    if extra_data is None or extra_keys is None:
        for pid, step, (x3, y3, z3), (x2, y2) in zip(ids, step_names, coords3, coords2):
            writer.writerow([pid, step, feat_dim_label, x3, y3, z3, x2, y2])
    else:
        for i, (pid, step, (x3, y3, z3), (x2, y2)) in enumerate(
            zip(ids, step_names, coords3, coords2)
        ):
            row = [pid, step, feat_dim_label, x3, y3, z3, x2, y2]
            for key in extra_keys:
                row.append(extra_data[key][i])
            writer.writerow(row)


def resolve_output_dir(output_dir: str) -> str:
    base_dir = folder_paths.get_output_directory() if folder_paths is not None else "outputs"
    if not output_dir:
        return base_dir
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(base_dir, output_dir)


def iter_csv_texts(
    csv_path: str,
    text_column: str,
    start: int,
    end: int,
    chunksize: int,
) -> Iterator[str]:
    """
    Stream prompt texts from CSV for the selected row range [start, end).
    end == -1 means until EOF.
    """
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
        for txt in sub[text_column].astype(str).tolist():
            yield txt

        seen += chunk_len


class UmapConfig:
    """
    Shared PCA/UMAP settings for downstream UMAP nodes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "use_pca": ("BOOLEAN", {"default": True, "tooltip": "Apply PCA before UMAP to reduce dimensionality and speed up fitting."}),
                "pca_dim": ("INT", {"default": 256, "min": 2, "max": 1024, "step": 1, "tooltip": "Target PCA dimension when use_pca is enabled."}),
                "use_umap_seed": ("BOOLEAN", {"default": True, "tooltip": "Enable deterministic UMAP initialization using umap_random_seed."}),
                "umap_random_seed": ("INT", {"default": 42, "min": 0, "max": 10_000, "tooltip": "Random seed used by UMAP when use_umap_seed is enabled."}),
                "n_neighbors": ("INT", {"default": 5, "min": 2, "max": 64, "tooltip": "UMAP neighborhood size. Smaller preserves local structure; larger smooths globally."}),
                "min_dist": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "UMAP minimum distance between embedded points."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("umap_config",)
    FUNCTION = "build"
    CATEGORY = "CLIP Intermediate Export"

    def build(
        self,
        use_pca: bool,
        pca_dim: int,
        use_umap_seed: bool,
        umap_random_seed: int,
        n_neighbors: int,
        min_dist: float,
    ):
        cfg = {
            "use_pca": use_pca,
            "pca_dim": pca_dim,
            "use_umap_seed": use_umap_seed,
            "umap_random_seed": umap_random_seed,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }
        return (cfg,)


class PromptUmapBuilder:
    """
    UMAP over prompt embeddings (per step only).
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "activations": ("MODEL", {"tooltip": "ActivationRun from CIE: Capture Activations."}),
                "dataset": ("MODEL", {"tooltip": "PromptDataset from CIE: Load Prompts CSV."}),
                "umap_config": ("MODEL", {"tooltip": "UMAP settings bundle from CIE: UMAP Config."}),
                "output_dir": ("STRING", {"default": "", "dir": True, "tooltip": "Output directory (absolute or relative to Comfy output folder)."}),
                "file_name": ("STRING", {"default": "clip_prompts_per_step_coords_3d2d.csv", "tooltip": "Output CSV filename for prompt coordinates."}),
                "write_lookup_table": ("BOOLEAN", {"default": False, "tooltip": "Also export clip_prompts_lookup.csv with id/source/prompt text."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coords_csv",)
    FUNCTION = "run"
    CATEGORY = "CLIP Intermediate Export"

    def run(
        self,
        activations: ActivationRun,
        dataset,
        umap_config: Dict[str, Any],
        output_dir: str,
        file_name: str,
        write_lookup_table: bool,
    ):
        resolved_dir = resolve_output_dir(output_dir)
        os.makedirs(resolved_dir, exist_ok=True)

        base_name = file_name.strip() or "clip_prompts_per_step_coords_3d2d.csv"
        coords_csv = os.path.join(resolved_dir, base_name)
        prompts_csv = os.path.join(resolved_dir, "clip_prompts_lookup.csv")

        if write_lookup_table:
            with open(prompts_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "source", "prompt"])
                texts = getattr(dataset, "texts", None)
                if texts is not None:
                    for pid, src, txt in zip(dataset.prompt_ids, dataset.sources, texts):
                        writer.writerow([pid, src, txt])
                else:
                    csv_path = str(getattr(dataset, "csv_path", "") or getattr(dataset, "path", ""))
                    text_column = str(getattr(dataset, "text_column", "text"))
                    start = int(getattr(dataset, "start", 0))
                    end = int(getattr(dataset, "end", -1))
                    chunksize = int(getattr(dataset, "chunksize", 200_000))
                    if not csv_path:
                        raise RuntimeError("Streaming dataset missing csv_path/path for lookup export.")
                    for pid, src, txt in zip(
                        dataset.prompt_ids,
                        dataset.sources,
                        iter_csv_texts(csv_path, text_column, start, end, chunksize),
                    ):
                        writer.writerow([pid, src, txt])

        cls_per_step = resolve_cls_per_step(activations)
        progress = ProgressBar(max(1, len(cls_per_step))) if ProgressBar is not None else None

        with open(coords_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "step", "feat_dim", "x3", "y3", "z3", "x2", "y2"])
            for step_name, X in cls_per_step.items():
                X_np = to_numpy_cpu(X)
                feat_dim = X_np.shape[1]
                steps = [step_name] * len(dataset.prompt_ids)
                run_umap_and_write(
                    X_np,
                    steps,
                    feat_dim,
                    writer,
                    dataset.prompt_ids,
                    umap_config["use_pca"],
                    umap_config["pca_dim"],
                    umap_config["use_umap_seed"],
                    umap_config["umap_random_seed"],
                    n_neighbors=umap_config["n_neighbors"],
                    min_dist=umap_config["min_dist"],
                )
                if progress is not None:
                    progress.update(1)

        return (coords_csv,)


class NeuronUmapBuilder:
    """
    UMAP over neuron activity (per step only).
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "activations": ("MODEL", {"tooltip": "ActivationRun from CIE: Capture Activations."}),
                "umap_config": ("MODEL", {"tooltip": "UMAP settings bundle from CIE: UMAP Config."}),
                "output_dir": ("STRING", {"default": "", "dir": True, "tooltip": "Output directory (absolute or relative to Comfy output folder)."}),
                "file_name": ("STRING", {"default": "clip_neurons_per_step_coords_3d2d.csv", "tooltip": "Output CSV filename for neuron coordinates."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coords_csv",)
    FUNCTION = "run"
    CATEGORY = "CLIP Intermediate Export"

    def run(
        self,
        activations: ActivationRun,
        umap_config: Dict[str, Any],
        output_dir: str,
        file_name: str,
    ):
        resolved_dir = resolve_output_dir(output_dir)
        os.makedirs(resolved_dir, exist_ok=True)

        base_name = file_name.strip() or "clip_neurons_per_step_coords_3d2d.csv"
        coords_csv = os.path.join(resolved_dir, base_name)

        cls_per_step = resolve_cls_per_step(activations)
        progress = ProgressBar(max(1, len(cls_per_step))) if ProgressBar is not None else None

        with open(coords_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "step",
                    "feat_dim",
                    "x3",
                    "y3",
                    "z3",
                    "x2",
                    "y2",
                    "act_mean",
                    "act_std",
                    "act_abs_mean",
                    "act_abs_max",
                    "sparsity",
                ]
            )
            for step_name, X in cls_per_step.items():
                X_np = to_numpy_cpu(X)
                feat_dim = X_np.shape[1]
                neuron_mat = X_np.T
                num_neurons = neuron_mat.shape[0]
                abs_acts = np.abs(neuron_mat)
                eps = 1e-3
                extra_data = {
                    "act_mean": neuron_mat.mean(axis=1),
                    "act_std": neuron_mat.std(axis=1),
                    "act_abs_mean": abs_acts.mean(axis=1),
                    "act_abs_max": abs_acts.max(axis=1),
                    "sparsity": (abs_acts > eps).mean(axis=1),
                }
                ids = [j for j in range(num_neurons)]
                steps = [step_name] * num_neurons
                run_umap_and_write(
                    neuron_mat,
                    steps,
                    feat_dim,
                    writer,
                    ids,
                    umap_config["use_pca"],
                    umap_config["pca_dim"],
                    umap_config["use_umap_seed"],
                    umap_config["umap_random_seed"],
                    n_neighbors=umap_config["n_neighbors"],
                    min_dist=umap_config["min_dist"],
                    extra_data=extra_data,
                    extra_keys=["act_mean", "act_std", "act_abs_mean", "act_abs_max", "sparsity"],
                )
                if progress is not None:
                    progress.update(1)

        return (coords_csv,)


NODE_CLASS_MAPPINGS = {
    "UmapConfig": UmapConfig,
    "PromptUmapBuilder": PromptUmapBuilder,
    "NeuronUmapBuilder": NeuronUmapBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UmapConfig": "CIE: UMAP Config",
    "PromptUmapBuilder": "CIE: UMAP - Prompts",
    "NeuronUmapBuilder": "CIE: UMAP - Neuron Activity",
}
