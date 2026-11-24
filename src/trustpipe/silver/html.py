"""Extract the main text of an HTML page.

Web pages carry navigation, footers, cookie banners and comment threads around the text
that matters. The main text is extracted with trafilatura, the extractor FineWeb uses for
its Common Crawl pipeline, with reader comments left out: they are user content with no
accountable publisher, whatever the trust tier of the page. Extraction is deterministic
for a given trafilatura version, which every silver receipt records.

Charset: a byte order mark or the HTTP header win (decoding.py); otherwise the charset a
``<meta>`` tag declares in the first 1024 bytes is used, as the WHATWG prescan does, and
UTF-8 when there is none.
"""

import re
import warnings
from functools import cache
from importlib.metadata import version
from typing import Protocol, cast

PLAIN_TYPES = frozenset({"text/plain", "text/markdown"})
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_PRESCAN_BYTES = 1024
_META_CHARSET = re.compile(rb"<meta[^>]*?charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE)


class _Extract(Protocol):
    def __call__(
        self,
        filecontent: str,
        *,
        include_comments: bool,
        include_tables: bool,
        deduplicate: bool,
        output_format: str,
    ) -> str | None: ...


def meta_charset(raw: bytes) -> str | None:
    """The charset label a ``<meta>`` tag declares near the start of the page, if any."""
    match = _META_CHARSET.search(raw[:_PRESCAN_BYTES])
    return match[1].decode("ascii").lower() if match else None


@cache
def _extract() -> _Extract:
    # Imported on first use: it is a large import, and its download helpers probe for
    # urllib3's optional SOCKS support, which warns when it is absent. Nothing here
    # downloads anything, so that warning is irrelevant and silenced at this one place.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
        import trafilatura

    return cast(_Extract, trafilatura.extract)


@cache
def extractor() -> str:
    """Name and version of the extractor, as recorded in silver receipts."""
    return f"trafilatura/{version('trafilatura')}"


def main_text(html: str) -> str | None:
    """The main text of a page, or None when the page has none."""
    return _extract()(
        html, include_comments=False, include_tables=True, deduplicate=False, output_format="txt"
    )
