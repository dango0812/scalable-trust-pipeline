from collections.abc import Callable
from pathlib import Path

import pytest
from corpus import AT, REGISTRY_TOML
from hypothesis import settings

from trustpipe.bronze.ingest import land
from trustpipe.lake.schema import Acquisition, RawCapture
from trustpipe.silver.refine import refine_all
from trustpipe.trust.registry import Registry, load_registry

# Property tests run the same examples on every machine and in CI, and leave no example
# database behind in the working tree.
settings.register_profile("repo", derandomize=True, database=None, deadline=None)
settings.load_profile("repo")


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    path = tmp_path / "sources.toml"
    path.write_text(REGISTRY_TOML, encoding="utf-8")
    return load_registry(path)


Page = tuple[str, str]  # (source URI, text)
Lander = Callable[[list[Page]], Path]


@pytest.fixture
def lake(tmp_path: Path, registry: Registry) -> Lander:
    """Land pages (one bronze part per call) and refine everything; return the lake root."""
    root = tmp_path / "lake"
    calls = [0]

    def land_pages(pages: list[Page]) -> Path:
        calls[0] += 1
        captures = [
            RawCapture.from_bytes(
                text.encode(),
                source_uri=uri,
                fetched_at=AT.replace(minute=calls[0], second=n),
                acquisition=Acquisition.HTTP_FETCH,
                content_type="text/plain; charset=utf-8",
            )
            for n, (uri, text) in enumerate(pages)
        ]
        land(captures, root)
        refine_all(root, registry)
        return root

    return land_pages
