"""Read and write web archives: WARC 1.0 and 1.1 (ISO 28500).

Crawling is not part of this pipeline. A crawler (the operator's own, or a public one such
as Common Crawl) hands over WARC files, the format web archives and crawlers share. A WARC
response record holds the exact HTTP response a server sent, the URI it came from and the
moment it was fetched, which is everything a bronze capture records. The pipeline itself
never touches the network, so the same files always land the same captures.

What a response record becomes:

- ``source_uri``: WARC-Target-URI in canonical form
- ``fetched_at``: WARC-Date
- ``http_status`` and ``content_type``: from the HTTP response
- ``raw``: the HTTP entity body with transfer and content codings removed (chunked, gzip,
  deflate), which is the document the server meant to send. Common Crawl stores bodies
  already decoded and renames the headers to ``X-Crawler-*``, which needs no decoding.

A record is skipped, with a reason that is counted, when it is not a response, is
truncated (``WARC-Truncated``), fails its ``WARC-Block-Digest``, is not a 2xx response,
uses a content coding this module does not decode, or is larger than the size limit (the
limit also bounds decompression, so a compression bomb cannot exhaust memory).

Gzip WARC files compress each record as its own gzip member; ``gzip`` reads them as one
stream, so compressed and plain files are read the same way.
"""

import base64
import gzip
import hashlib
import io
import uuid
import zlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, cast

from trustpipe.lake.schema import Acquisition, RawCapture

VERSIONS = (b"WARC/1.0", b"WARC/1.1")
MAX_RECORD_BYTES = 64 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"
_CRLF = b"\r\n"
_DIGESTS = {"sha1": "sha1", "sha256": "sha256", "sha512": "sha512", "md5": "md5"}


class WarcFormatError(ValueError):
    """The file is not a well-formed WARC stream; nothing after the error can be trusted."""


@dataclass(frozen=True, slots=True)
class WarcRecord:
    """One record: its header fields (names lower-cased) and its content block.

    ``block`` is None when the block was larger than the reader's limit; it was skipped
    without being held in memory.
    """

    version: str
    headers: Mapping[str, str]
    block: bytes | None

    @property
    def type(self) -> str:
        return self.headers.get("warc-type", "")


@contextmanager
def open_archive(path: Path) -> Iterator[BinaryIO]:
    """Open a WARC file for reading, decompressing it when it is gzip-compressed."""
    with path.open("rb") as raw:
        compressed = raw.read(2) == _GZIP_MAGIC
        raw.seek(0)
        if not compressed:
            yield cast(BinaryIO, raw)
            return
        with gzip.GzipFile(fileobj=raw, mode="rb") as unzipped:
            yield cast(BinaryIO, unzipped)


def _header_fields(stream: BinaryIO) -> dict[str, str]:
    fields: dict[str, str] = {}
    last = ""
    while True:
        line = stream.readline()
        if not line:
            raise WarcFormatError("the stream ends inside a record header")
        if line in (_CRLF, b"\n"):
            return fields
        text = line.decode("utf-8").rstrip("\r\n")
        if text[:1] in (" ", "\t") and last:  # a folded continuation of the previous field
            fields[last] += " " + text.strip()
            continue
        name, colon, value = text.partition(":")
        if not colon:
            raise WarcFormatError(f"header line without a colon: {text!r}")
        last = name.strip().lower()
        fields[last] = value.strip()


def _skip(stream: BinaryIO, length: int) -> None:
    while length > 0:
        chunk = stream.read(min(length, 1 << 20))
        if not chunk:
            raise WarcFormatError("the stream ends inside a record block")
        length -= len(chunk)


def iter_records(
    stream: BinaryIO, *, max_block_bytes: int = MAX_RECORD_BYTES
) -> Iterator[WarcRecord]:
    """Yield every record of a (decompressed) WARC stream in order."""
    while True:
        line = stream.readline()
        if not line:
            return
        if line in (_CRLF, b"\n"):  # tolerate extra blank lines between records
            continue
        version = line.rstrip(b"\r\n")
        if version not in VERSIONS:
            raise WarcFormatError(f"expected a WARC/1.0 or WARC/1.1 record, got {line[:40]!r}")
        headers = _header_fields(stream)
        try:
            length = int(headers["content-length"])
        except (KeyError, ValueError):
            raise WarcFormatError("a record needs a numeric Content-Length") from None
        if length < 0:
            raise WarcFormatError("Content-Length must not be negative")
        block: bytes | None
        if length > max_block_bytes:
            _skip(stream, length)
            block = None
        else:
            block = stream.read(length)
            if len(block) != length:
                raise WarcFormatError("the stream ends inside a record block")
        yield WarcRecord(version.decode("ascii"), headers, block)


def digest_matches(value: str, data: bytes) -> bool | None:
    """Check a WARC digest field ("sha1:BASE32" or hex); None if the algorithm is unknown."""
    algorithm, colon, encoded = value.partition(":")
    name = _DIGESTS.get(algorithm.strip().lower())
    if not colon or name is None:
        return None
    digest = hashlib.new(name, data).digest()
    encoded = encoded.strip()
    if len(encoded) == 2 * len(digest):
        return encoded.lower() == digest.hex()
    return encoded.upper().rstrip("=") == base64.b32encode(digest).decode("ascii").rstrip("=")


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # names lower-cased; repeated fields joined with ", "
    body: bytes  # transfer and content codings removed


class Skip(Exception):  # noqa: N818 - a control-flow signal carrying a reason code, not an error
    """A record that is valid WARC but does not become a capture."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _dechunk(body: bytes) -> bytes:
    """Remove the chunked transfer coding (RFC 9112 section 7.1)."""
    out = bytearray()
    view = io.BytesIO(body)
    while True:
        size_line = view.readline()
        if not size_line:
            raise Skip("malformed_http", "chunked body ends without a last chunk")
        try:
            size = int(size_line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            raise Skip("malformed_http", f"bad chunk size {size_line[:20]!r}") from None
        if size == 0:
            return bytes(out)
        chunk = view.read(size)
        if len(chunk) != size:
            raise Skip("malformed_http", "chunked body ends inside a chunk")
        out += chunk
        view.readline()  # the CRLF after each chunk


def _decompress(body: bytes, wbits: int, limit: int) -> bytes:
    decoder = zlib.decompressobj(wbits)
    try:
        data = decoder.decompress(body, limit + 1)
    except zlib.error as error:
        raise Skip("malformed_http", f"content coding does not decode: {error}") from None
    if len(data) > limit:
        raise Skip("oversized", f"decoded body exceeds {limit} bytes")
    return data


def _decode_content(body: bytes, coding: str, limit: int) -> bytes:
    for name in (c.strip().lower() for c in reversed(coding.split(","))):
        if name in ("", "identity"):
            continue
        if name in ("gzip", "x-gzip"):
            body = _decompress(body, 16 + zlib.MAX_WBITS, limit)
        elif name == "deflate":
            # RFC 9110 says zlib-wrapped; many servers send raw deflate, so accept both.
            wrapped = body[:1] == b"\x78"
            body = _decompress(body, zlib.MAX_WBITS if wrapped else -zlib.MAX_WBITS, limit)
        else:
            raise Skip("unsupported_content_encoding", name)
    return body


def parse_http_response(block: bytes, *, max_body_bytes: int = MAX_RECORD_BYTES) -> HttpResponse:
    """Split an HTTP/1.x response into status, headers and decoded entity body."""
    head, separator, body = block.partition(b"\r\n\r\n")
    if not separator:
        head, separator, body = block.partition(b"\n\n")
    if not separator:
        raise Skip("malformed_http", "no blank line after the response header")
    lines = head.decode("iso-8859-1").splitlines()
    status_line = lines[0].split(" ", 2)
    if len(status_line) < 2 or not status_line[0].startswith("HTTP/"):
        raise Skip("malformed_http", f"bad status line {lines[0][:60]!r}")
    try:
        status = int(status_line[1])
    except ValueError:
        raise Skip("malformed_http", f"bad status code {status_line[1][:10]!r}") from None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, colon, value = line.partition(":")
        if colon:
            key = name.strip().lower()
            headers[key] = f"{headers[key]}, {value.strip()}" if key in headers else value.strip()
    if "chunked" in headers.get("transfer-encoding", "").lower():
        body = _dechunk(body)
    body = _decode_content(body, headers.get("content-encoding", ""), max_body_bytes)
    return HttpResponse(status, headers, body)


def _warc_date(value: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise Skip("invalid_date", value[:40]) from None
    if moment.tzinfo is None:
        raise Skip("invalid_date", f"{value[:40]} has no UTC designator")
    return moment.astimezone(UTC)


def capture_from_record(record: WarcRecord, acquisition: Acquisition) -> RawCapture:
    """Turn one response record into a capture, or raise ``Skip`` with the reason."""
    if record.type != "response":
        raise Skip(f"record_{record.type or 'untyped'}")
    if record.block is None:
        raise Skip("oversized", "record block exceeds the size limit")
    if "warc-truncated" in record.headers:
        raise Skip("truncated", record.headers["warc-truncated"])
    block_digest = record.headers.get("warc-block-digest")
    if block_digest is not None and digest_matches(block_digest, record.block) is False:
        raise Skip("block_digest_mismatch", block_digest)
    fetched_at = _warc_date(record.headers.get("warc-date", ""))
    response = parse_http_response(record.block)
    if not 200 <= response.status < 300:
        raise Skip("http_status", str(response.status))
    try:
        return RawCapture.from_bytes(
            response.body,
            source_uri=record.headers.get("warc-target-uri", "").strip("<>"),
            fetched_at=fetched_at,
            acquisition=acquisition,
            http_status=response.status,
            content_type=response.headers.get("content-type"),
        )
    except ValueError as error:
        raise Skip("invalid_uri", str(error)[:200]) from None


def response_record(
    uri: str,
    fetched_at: datetime,
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "text/html; charset=utf-8",
) -> bytes:
    """One gzip-compressed WARC/1.1 response record, the unit a crawler appends to a file.

    For crawlers written in Python that want to hand their output to this pipeline; any
    WARC writer (warcio, Heritrix, wget --warc-file) produces files that read the same.
    """
    http = (
        f"HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode("iso-8859-1") + body
    digest = base64.b32encode(hashlib.sha1(http, usedforsecurity=False).digest()).decode()
    moment = fetched_at.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    header = (
        "WARC/1.1\r\nWARC-Type: response\r\n"
        f"WARC-Target-URI: {uri}\r\nWARC-Date: {moment}\r\n"
        f"WARC-Record-ID: <urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, uri + moment + digest)}>\r\n"
        "Content-Type: application/http; msgtype=response\r\n"
        f"WARC-Block-Digest: sha1:{digest}\r\nContent-Length: {len(http)}\r\n\r\n"
    ).encode()
    return gzip.compress(header + http + b"\r\n\r\n", mtime=0)
