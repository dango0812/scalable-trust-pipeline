"""Commit records: the outputs of a part count only once its receipt exists.

A stage writes every output file of a part first and the receipt last. A crash at any
moment therefore leaves either a committed part (receipt present) or an uncommitted one
that the next run redoes; readers and resume decide by receipts, never by loose files.

A receipt names the stage, the input files, every output file with its row count, size and
SHA-256, and the transform that produced them (package version, canonical text form,
policy). It holds no timestamps, so re-running a deterministic stage reproduces the same
receipt byte for byte; times belong to the run log. The receipt is what ``verify``
re-checks, and the lineage of every file is in it.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from trustpipe.lake.layout import receipt_key
from trustpipe.lake.storage import WrittenFile, lake_path, write_json

FORMAT = "trustpipe.receipt/1"


@dataclass(frozen=True, slots=True)
class Receipt:
    stage: str
    part_id: str
    inputs: tuple[str, ...]
    outputs: tuple[WrittenFile, ...]
    transform: dict[str, str]
    counters: dict[str, int]

    @property
    def key(self) -> str:
        return receipt_key(self.stage, self.part_id)

    def document(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "stage": self.stage,
            "part_id": self.part_id,
            "inputs": list(self.inputs),
            "outputs": [
                {"key": f.key, "rows": f.rows, "bytes": f.bytes, "sha256": f.sha256}
                for f in self.outputs
            ],
            "transform": self.transform,
            "counters": self.counters,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> Self:
        if document.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} document: {document.get('format')!r}")
        return cls(
            stage=document["stage"],
            part_id=document["part_id"],
            inputs=tuple(document["inputs"]),
            outputs=tuple(
                WrittenFile(o["key"], o["bytes"], o["sha256"], o["rows"])
                for o in document["outputs"]
            ),
            transform=dict(document["transform"]),
            counters=dict(document["counters"]),
        )


def commit(root: Path, receipt: Receipt, *, create_only: bool) -> WrittenFile:
    """Publish the receipt; from this moment the part's outputs are visible."""
    return write_json(root, receipt.key, receipt.document(), create_only=create_only)


def is_committed(root: Path, stage: str, part: str) -> bool:
    return lake_path(root, receipt_key(stage, part)).is_file()


def load_receipt(root: Path, stage: str, part: str) -> Receipt:
    path = lake_path(root, receipt_key(stage, part))
    return Receipt.from_document(json.loads(path.read_text(encoding="utf-8")))


def committed_parts(root: Path, stage: str) -> list[str]:
    """Part ids with a receipt for ``stage``, sorted."""
    directory = lake_path(root, receipt_key(stage, "0" * 16)).parent
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))
