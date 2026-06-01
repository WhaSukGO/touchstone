"""CUDA image registry (design §4.3).

Frameworks map to PRE-BUILT images via a curated matrix. Runtime image builds are
forbidden — that was a resource sink in the prior attempt. Resolution is pure (no
Docker needed), so it is unit-testable without a GPU."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import FrameworkSpec


class NoImageError(RuntimeError):
    pass


@dataclass
class ResolvedImage:
    key: str
    image: str
    cuda: str
    healthcheck: str | None = None
    digest: str | None = None


class ImageRegistry:
    def __init__(self, matrix_path: str | Path):
        self.matrix_path = Path(matrix_path)
        self._images: list[ResolvedImage] = []
        if self.matrix_path.exists():
            self._load()

    def _load(self) -> None:
        doc = yaml.safe_load(self.matrix_path.read_text()) or {}
        self._images = [
            ResolvedImage(
                key=e["key"], image=e["image"], cuda=str(e.get("cuda", "")),
                healthcheck=e.get("healthcheck"), digest=e.get("digest"),
            )
            for e in doc.get("images", [])
        ]

    def list_matrix(self) -> list[ResolvedImage]:
        return list(self._images)

    def resolve(self, fw: FrameworkSpec) -> ResolvedImage:
        """Best match by (name, version prefix, cuda). Raises if nothing pre-built fits."""
        candidates = [
            img for img in self._images
            if fw.name in img.key
            and (not fw.version or fw.version in img.key)
            and (not fw.cuda or img.cuda == fw.cuda)
        ]
        if not candidates:
            raise NoImageError(
                f"no prebuilt image for framework={fw.name} {fw.version} cuda={fw.cuda}; "
                f"add it to {self.matrix_path} (runtime builds are disabled)"
            )
        # Prefer exact cuda match, then most specific (longest key).
        candidates.sort(key=lambda i: (i.cuda == fw.cuda, len(i.key)), reverse=True)
        return candidates[0]
