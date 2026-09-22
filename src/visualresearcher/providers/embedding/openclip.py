"""CLIP embeddings via ``open_clip`` (CLAUDE.md §4: ViT-B-32, laion2b_s34b_b79k).

Imported lazily so ``doctor`` can report it missing without the process dying,
and so ``VR_OFFLINE=1`` never needs torch at all.

Two environment rules from §3 are enforced here rather than assumed:

* weights follow ``HF_HOME``; no cache path is hardcoded,
* the device defaults to CPU. ``device: auto`` picks CUDA only when torch
  actually reports a working CUDA device, because §3 says CPU-only unless the
  user confirms a GPU.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ...logging_setup import get_logger
from ..base import Availability, ProviderError
from .base import EmbeddingProvider

__all__ = ["OpenClipEmbeddingProvider"]

log = get_logger("providers.embedding.openclip")

#: How many images to push through the model at once on CPU.
BATCH = 16


class OpenClipEmbeddingProvider(EmbeddingProvider):
    name = "openclip"
    meaningful = True

    def __init__(
        self,
        *,
        model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        seed: int = 1729,
        **_ignored,
    ) -> None:
        self.model_name = model
        self.pretrained = pretrained
        self.requested_device = device
        self.seed = seed
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    # -- readiness ---------------------------------------------------------

    def availability(self) -> Availability:
        missing = [
            name for name in ("torch", "open_clip") if importlib.util.find_spec(name) is None
        ]
        if missing:
            return Availability.unavailable(
                f"{', '.join(missing)} not installed",
                "uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
                "uv pip install open_clip_torch",
            )
        if not os.environ.get("HF_HOME"):
            return Availability.available(
                f"{self.model_name}/{self.pretrained} "
                "(warning: HF_HOME unset; weights may land on C:)"
            )
        return Availability.available(
            f"{self.model_name}/{self.pretrained} on {self._resolve_device()}"
        )

    def _resolve_device(self) -> str:
        if self.requested_device and self.requested_device != "auto":
            return self.requested_device
        try:
            import torch

            # §3: CPU unless a GPU is genuinely present and usable.
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    # -- model -------------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model

        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by availability()
            raise ProviderError(f"open_clip/torch not installed: {exc}") from exc

        # §11: deterministic under a fixed seed.
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(False)  # CLIP inference is already deterministic

        self._device = self._resolve_device()
        log.info(
            "loading CLIP %s/%s on %s (weights follow HF_HOME=%s)",
            self.model_name,
            self.pretrained,
            self._device,
            os.environ.get("HF_HOME", "<unset>"),
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self._device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self.dimensions = int(model.visual.output_dim)
        return self._model

    # -- embedding ---------------------------------------------------------

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        import torch
        from PIL import Image

        model = self._load()
        out: list[list[float]] = [[] for _ in paths]

        pending: list[int] = []
        tensors: list = []
        for position, path in enumerate(paths):
            try:
                with Image.open(path) as source:
                    tensors.append(self._preprocess(source.convert("RGB")))
                pending.append(position)
            except Exception as exc:  # noqa: BLE001 - one bad file, zero vector
                log.debug("could not embed %s: %s", path, exc)
                out[position] = [0.0] * self.dimensions

        for start in range(0, len(tensors), BATCH):
            chunk = tensors[start : start + BATCH]
            positions = pending[start : start + BATCH]
            with torch.no_grad():
                batch = torch.stack(chunk).to(self._device)
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            for position, vector in zip(positions, features.cpu().tolist(), strict=True):
                out[position] = vector

        for position, vector in enumerate(out):
            if not vector:
                out[position] = [0.0] * self.dimensions
        return out

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        import torch

        model = self._load()
        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self._device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()
