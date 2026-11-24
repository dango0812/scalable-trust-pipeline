import base64
import gzip
import hashlib
import io
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustpipe.archive.warc import (
    Skip,
    WarcFormatError,
    WarcRecord,
    capture_from_record,
    digest_matches,
    iter_records,
    open_archive,
    parse_http_response,
    response_record,
)
from trustpipe.lake.schema import Acquisition

AT = datetime(2026, 11, 24, 8, 0, tzinfo=UTC)
PAGE = b"<html><body><p>hello</p></body></html>"


def warc(headers: dict[str, str], block: bytes, version: bytes = b"WARC/1.1") -> bytes:
    lines = b"".join(f"{k}: {v}\r\n".encode() for k, v in headers.items())
    return version + b"\r\n" + lines + b"\r\n" + block + b"\r\n\r\n"


def http(body: bytes, *headers: str, status: str = "200 OK") -> bytes:
    head = "".join(f"{h}\r\n" for h in headers)
    return f"HTTP/1.1 {status}\r\n{head}\r\n".encode() + body


def response(block: bytes, **extra: str) -> WarcRecord:
    headers = {
        "warc-type": "response",
        "warc-target-uri": "https://example.org/page",
        "warc-date": "2026-11-24T08:00:00Z",
        "content-length": str(len(block)),
        **extra,
    }
    return WarcRecord("WARC/1.1", headers, block)


def records(data: bytes, **options: int) -> list[WarcRecord]:
    return list(iter_records(io.BytesIO(data), **options))


def test_a_written_record_reads_back_as_the_same_capture(tmp_path: Path) -> None:
    path = tmp_path / "crawl.warc.gz"
    path.write_bytes(response_record("https://Example.org/a", AT, PAGE))
    with open_archive(path) as stream:
        [record] = list(iter_records(stream))
    capture = capture_from_record(record, Acquisition.HTTP_FETCH)
    assert capture.source_uri == "https://example.org/a"
    assert capture.fetched_at == AT
    assert capture.raw == PAGE
    assert (capture.http_status, capture.content_type) == (200, "text/html; charset=utf-8")
    assert capture.acquisition is Acquisition.HTTP_FETCH


def test_plain_files_and_extra_blank_lines_are_read(tmp_path: Path) -> None:
    path = tmp_path / "crawl.warc"
    info = warc({"WARC-Type": "warcinfo", "Content-Length": "2"}, b"hi", version=b"WARC/1.0")
    path.write_bytes(b"\r\n" + info + b"\n" + warc({"Content-Length": "0"}, b""))
    with open_archive(path) as stream:
        found = list(iter_records(stream))
    assert [(r.version, r.type, r.block) for r in found] == [
        ("WARC/1.0", "warcinfo", b"hi"),
        ("WARC/1.1", "", b""),
    ]


def test_folded_header_lines_continue_the_previous_field() -> None:
    data = b"WARC/1.1\r\nWARC-Type: resp\r\n onse\r\nContent-Length: 0\r\n\r\n"
    [record] = records(data)
    assert record.type == "resp onse"


def test_a_header_that_starts_folded_is_read_as_a_field() -> None:
    [record] = records(b"WARC/1.1\r\n Content-Length: 0\r\n\r\n")
    assert record.headers == {"content-length": "0"}


def test_oversized_blocks_are_skipped_without_being_held() -> None:
    data = warc({"Content-Length": "10"}, b"0123456789") + warc({"Content-Length": "1"}, b"x")
    assert [r.block for r in records(data, max_block_bytes=5)] == [None, b"x"]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"HTTP/1.1 200 OK\r\n", "expected a WARC/1.0 or WARC/1.1 record"),
        (b"WARC/1.1\r\nWARC-Type: x\r\n", "ends inside a record header"),
        (b"WARC/1.1\r\nno colon here\r\n\r\n", "header line without a colon"),
        (b"WARC/1.1\r\nWARC-Type: x\r\n\r\n", "numeric Content-Length"),
        (b"WARC/1.1\r\nContent-Length: ten\r\n\r\n", "numeric Content-Length"),
        (b"WARC/1.1\r\nContent-Length: -1\r\n\r\n", "must not be negative"),
        (b"WARC/1.1\r\nContent-Length: 9\r\n\r\nshort", "ends inside a record block"),
    ],
)
def test_malformed_streams_are_refused(data: bytes, message: str) -> None:
    with pytest.raises(WarcFormatError, match=message):
        records(data)


def test_an_oversized_block_that_is_cut_off_is_refused() -> None:
    with pytest.raises(WarcFormatError, match="ends inside a record block"):
        records(b"WARC/1.1\r\nContent-Length: 9\r\n\r\nshort", max_block_bytes=2)


def test_digests_in_base32_and_hex() -> None:
    data = b"block"
    sha1 = hashlib.sha1(data, usedforsecurity=False).digest()
    assert digest_matches("sha1:" + base64.b32encode(sha1).decode(), data) is True
    assert digest_matches("SHA1:" + base64.b32encode(sha1).decode().lower(), data) is True
    assert digest_matches("sha256:" + hashlib.sha256(data).hexdigest(), data) is True
    assert digest_matches("sha256:" + hashlib.sha256(b"other").hexdigest(), data) is False
    assert digest_matches("sha1:AAAA", data) is False
    assert digest_matches("crc32:1234", data) is None
    assert digest_matches("no-algorithm", data) is None


def test_chunked_and_compressed_bodies_are_decoded() -> None:
    chunked = b"5;ext=1\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
    parsed = parse_http_response(http(chunked, "Transfer-Encoding: chunked"))
    assert parsed.body == b"hello world"
    for coding, body in [
        ("gzip", gzip.compress(PAGE)),
        ("x-gzip", gzip.compress(PAGE)),
        ("deflate", zlib.compress(PAGE)),
        ("deflate", zlib.compress(PAGE)[2:-4]),  # raw deflate, as many servers send it
        ("identity", PAGE),
        ("deflate, gzip", gzip.compress(zlib.compress(PAGE))),
    ]:
        assert parse_http_response(http(body, f"Content-Encoding: {coding}")).body == PAGE


def test_response_headers_are_case_insensitive_and_repeats_are_joined() -> None:
    parsed = parse_http_response(http(b"x", "Set-Cookie: a", "set-cookie: b", "no-colon"))
    assert parsed.headers == {"set-cookie": "a, b"}
    assert parse_http_response(b"HTTP/1.0 204 No Content\n\n").status == 204


@pytest.mark.parametrize(
    ("block", "reason"),
    [
        (b"HTTP/1.1 200 OK\r\nno end of header", "malformed_http"),
        (b"FTP 200 OK\r\n\r\n", "malformed_http"),
        (b"HTTP/1.1 OK\r\n\r\n", "malformed_http"),
        (http(b"5\r\nhel", "Transfer-Encoding: chunked"), "malformed_http"),
        (http(b"zz\r\n", "Transfer-Encoding: chunked"), "malformed_http"),
        (http(b"3\r\nabc\r\n", "Transfer-Encoding: chunked"), "malformed_http"),
        (http(b"not gzip", "Content-Encoding: gzip"), "malformed_http"),
        (http(PAGE, "Content-Encoding: br"), "unsupported_content_encoding"),
    ],
)
def test_bad_http_responses_are_skipped_with_a_reason(block: bytes, reason: str) -> None:
    with pytest.raises(Skip) as raised:
        parse_http_response(block)
    assert raised.value.reason == reason


def test_decompression_is_bounded() -> None:
    bomb = gzip.compress(b"\0" * 10_000)
    with pytest.raises(Skip, match="oversized"):
        parse_http_response(http(bomb, "Content-Encoding: gzip"), max_body_bytes=1_000)


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (WarcRecord("WARC/1.1", {"warc-type": "request"}, b""), "record_request"),
        (WarcRecord("WARC/1.1", {}, b""), "record_untyped"),
        (WarcRecord("WARC/1.1", {"warc-type": "response"}, None), "oversized"),
        (response(http(PAGE), **{"warc-truncated": "length"}), "truncated"),
        (response(http(PAGE), **{"warc-block-digest": "sha1:AAAA"}), "block_digest_mismatch"),
        (response(http(PAGE), **{"warc-date": "yesterday"}), "invalid_date"),
        (response(http(PAGE), **{"warc-date": "2026-11-24T08:00:00"}), "invalid_date"),
        (response(http(PAGE, status="404 Not Found")), "http_status"),
        (response(http(PAGE), **{"warc-target-uri": "ftp://example.org/"}), "invalid_uri"),
    ],
)
def test_records_that_do_not_become_captures(record: WarcRecord, reason: str) -> None:
    with pytest.raises(Skip) as raised:
        capture_from_record(record, Acquisition.ARCHIVE_IMPORT)
    assert raised.value.reason == reason
    assert reason in str(raised.value)


def test_unknown_digest_algorithms_and_bracketed_uris_are_accepted() -> None:
    record = response(
        http(PAGE, "Content-Type: text/html"),
        **{"warc-block-digest": "crc32:00", "warc-target-uri": "<https://example.org/x>"},
    )
    capture = capture_from_record(record, Acquisition.ARCHIVE_IMPORT)
    assert capture.source_uri == "https://example.org/x"
    assert capture.content_type == "text/html"
