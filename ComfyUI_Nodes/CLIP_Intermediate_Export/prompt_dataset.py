from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


REQS = ["pandas>=2.0.0"]


@dataclass
class PromptDataset:
    texts: Optional[List[str]]
    prompt_ids: List[Any]
    sources: List[Any]
    path: str
    load_mode: str = "memory"  # 'memory' | 'stream'
    csv_path: str = ""
    text_column: str = "text"
    id_column: str = "prompt_id"
    source_column: str = "source"
    start: int = 0
    end: int = -1
    chunksize: int = 200000


class PromptDatasetLoader:
    """
    Loads prompt metadata from CSV.

    Modes:
    - memory: load selected prompt texts into RAM (best for smaller datasets).
    - stream: keep only metadata in RAM; capture node streams texts directly from CSV.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "csv_path": ("STRING", {"default": "datasets/prompts_dataset.csv", "file": True,
                                        "tooltip": "Input CSV containing prompts and optional metadata columns."}),
                "load_mode": (["memory", "stream"], {"default": "memory",
                                                     "tooltip": "memory: load prompt texts into RAM. stream: keep RAM low and let capture stream texts from CSV."}),
                "start": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "step": 1,
                                  "tooltip": "Start row index (inclusive)."}),
                "end": ("INT", {"default": -1, "min": -1, "max": 10_000_000, "step": 1,
                                "tooltip": "End row index (exclusive). Use -1 for all remaining rows."}),
                "text_column": ("STRING", {"default": "text",
                                           "tooltip": "CSV column containing prompt text."}),
                "id_column": ("STRING", {"default": "prompt_id",
                                         "tooltip": "Optional ID column for prompts. Fallback IDs are generated if missing."}),
                "source_column": ("STRING", {"default": "source",
                                             "tooltip": "Optional source/domain column for prompts."}),
                "chunksize": ("INT", {"default": 200_000, "min": 1_000, "max": 2_000_000, "step": 1_000,
                                      "tooltip": "CSV chunk size for loading/metadata pass. Larger is faster, smaller uses less RAM."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("dataset",)
    FUNCTION = "load_dataset"
    CATEGORY = "CLIP Intermediate Export"

    def load_dataset(
        self,
        csv_path: str,
        load_mode: str,
        text_column: str,
        id_column: str,
        source_column: str,
        start: int,
        end: int,
        chunksize: int,
    ) -> Tuple[PromptDataset]:
        csv_file = Path(csv_path)

        mode = (load_mode or "memory").strip().lower()
        if mode not in {"memory", "stream"}:
            raise ValueError("load_mode must be 'memory' or 'stream'.")

        # In stream mode we avoid loading text payloads into RAM.
        usecols = []
        if mode == "memory":
            usecols.append(text_column)
        if id_column:
            usecols.append(id_column)
        if source_column:
            usecols.append(source_column)
        if not usecols:
            usecols = [text_column]

        texts: Optional[List[str]] = [] if mode == "memory" else None
        prompt_ids: List[Any] = []
        sources: List[Any] = []

        start = max(0, int(start))
        end = int(end)

        # Iterate chunks and keep only the requested slice.
        seen = 0
        for chunk in pd.read_csv(csv_file, usecols=lambda c: c in usecols, chunksize=int(chunksize)):
            if mode == "memory" and text_column not in chunk.columns:
                raise ValueError(f"Missing required column '{text_column}' in {csv_file}")

            chunk_len = len(chunk)
            chunk_start = seen
            chunk_end = seen + chunk_len

            # No overlap with desired range
            desired_end = end if end != -1 else float("inf")
            if chunk_end <= start:
                seen += chunk_len
                continue
            if chunk_start >= desired_end:
                break

            # Overlap slice indices inside this chunk
            take_from = max(0, start - chunk_start)
            take_to = chunk_len if end == -1 else min(chunk_len, end - chunk_start)

            sub = chunk.iloc[take_from:take_to]

            if mode == "memory":
                assert texts is not None
                texts.extend(sub[text_column].astype(str).tolist())

            if id_column in sub.columns:
                prompt_ids.extend(sub[id_column].tolist())
            else:
                # Fallback IDs for this slice
                base = start + len(prompt_ids)
                prompt_ids.extend(list(range(base, base + len(sub))))

            if source_column in sub.columns:
                sources.extend(sub[source_column].tolist())
            else:
                sources.extend(["unknown"] * len(sub))

            seen += chunk_len

        if not prompt_ids:
            raise ValueError("No rows loaded. Check start/end ranges and CSV path.")
        if mode == "memory" and not texts:
            raise ValueError("No prompt texts loaded. Check text_column/start/end.")

        dataset = PromptDataset(
            texts=texts,
            prompt_ids=prompt_ids,
            sources=sources,
            path=str(csv_file),
            load_mode=mode,
            csv_path=str(csv_file),
            text_column=text_column,
            id_column=id_column,
            source_column=source_column,
            start=start,
            end=end,
            chunksize=int(chunksize),
        )
        return (dataset,)


NODE_CLASS_MAPPINGS = {
    "PromptDatasetLoader": PromptDatasetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDatasetLoader": "CIE: Load Prompts CSV",
}
