"""Serial, rate-limit-aware preparation of the original Sen calibration videos.

This helper is CPU-only and uses Python's standard library. It never changes the
frozen sample manifest, rotates hosts, or replaces an existing source video.
"""
from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


USER_AGENT = "CE6190-Calibration-Recovery/1.0 (https://github.com/AvianHan/CE6190-Visual-Generation)"
MAX_ATTEMPTS = 8
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
CHUNK_BYTES = 1024 * 1024


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n"); output.flush(); os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry_after_seconds(value, now):
    """Parse the standard delta-seconds or HTTP-date Retry-After forms."""
    if value is None: return None
    value = str(value).strip()
    if value.isdigit(): return float(value)
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=dt.timezone.utc)
        seconds = parsed.timestamp() - now
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (TypeError, ValueError, OverflowError):
        return None


def retry_delay(attempt, headers, now):
    backoff = float(min(3600, 60 * 2 ** (attempt - 1)))
    supplied = retry_after_seconds(headers.get("Retry-After") if headers else None, now)
    # The exponential component is capped; a longer server delay is respected.
    return max(backoff, supplied or 0.0)


def _video_path(run_dir, sample):
    root = Path(run_dir).resolve() / "calibration"
    value = sample.get("video_path")
    if not isinstance(value, str) or not value.strip(): raise ValueError("Missing video_path")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    if path.is_symlink(): raise ValueError(f"Source video must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_relative_to((root / "videos").resolve()):
        raise ValueError(f"Source video escapes calibration/videos: {path}")
    return path


def _valid_video_file(path, expected, *, source_url, status):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError(f"Not a regular source video: {path}")
    size = path.stat().st_size
    if size < 1: raise ValueError(f"Empty source video: {path}")
    with path.open("rb") as source: prefix = source.read(512)
    stripped = prefix.lstrip()
    if stripped.startswith((b"<", b"{", b"[")):
        raise ValueError(f"Source video contains an HTML/JSON document: {path}")
    if path.suffix.lower() == ".webm" and not prefix.startswith(b"\x1aE\xdf\xa3"):
        raise ValueError(f"Source video does not have a WebM/EBML header: {path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Source video checksum mismatch: {path}")
    return {"path": str(path), "source_url": source_url, "status": status,
            "bytes": size, "sha256": actual, "expected_sha256": expected,
            "checksum_status": "matches_expected" if expected else "observed_only"}


def _download_once(url, path, expected, opener):
    """Download to a temporary sibling, then publish without replacing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".recovery-download-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "video/*,application/octet-stream;q=0.9"})
            with opener(request, timeout=60) as response:
                final = urlsplit(response.geturl())
                if final.scheme != "https" or not final.netloc or final.username or final.password:
                    raise ValueError("Source video redirected away from authenticated-free HTTPS")
                kind = response.headers.get("Content-Type", "").lower()
                if "html" in kind or "json" in kind or kind.startswith("text/"):
                    raise ValueError("Source URL returned a document instead of video bytes")
                digest = hashlib.sha256(); size = 0
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                    target.write(chunk); digest.update(chunk); size += len(chunk)
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and int(content_length) != size:
                    raise URLError("Incomplete source video transfer")
                if size == 0: raise ValueError("Empty source video download")
                if expected is not None and digest.hexdigest() != expected:
                    raise ValueError("Downloaded source video checksum mismatch")
            target.flush(); os.fsync(target.fileno())
        # Check the container prefix before installing the final destination.
        with temporary.open("rb") as source: prefix = source.read(512)
        if prefix.lstrip().startswith((b"<", b"{", b"[")):
            raise ValueError("Downloaded source is an HTML/JSON document")
        if path.suffix.lower() == ".webm" and not prefix.startswith(b"\x1aE\xdf\xa3"):
            raise ValueError("Downloaded source does not have a WebM/EBML header")
        try:
            os.link(temporary, path)
            status = "downloaded"
        except FileExistsError:
            status = "existing_concurrent"
        record = _valid_video_file(path, expected, source_url=url, status=status)
        if status == "existing_concurrent" and record["sha256"] != digest.hexdigest():
            raise ValueError("A concurrently published source differs; preserved it for inspection")
        record["final_url"] = response.geturl()
        return record
    finally:
        temporary.unlink(missing_ok=True)


def prepare_assets(run_dir, audit_path=None, progress_callback=None, *, indices=(0, 2, 7),
                   sleep=None, now=None, opener=None, max_attempts=MAX_ATTEMPTS):
    """Verify selected original videos and serially download only missing ones.

    ``audit_path`` may be an explicit JSON path or an audit directory. Progress
    events are persisted before a callback is invoked. No source is overwritten.
    The default eight attempts bound network retries, not GPU concurrency/time.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    sleep = sleep or time.sleep; now = now or time.time; opener = opener or urlopen
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError("max_attempts must be between 1 and 8")
    source = run_dir / "calibration/samples.jsonl"
    frozen = source.read_bytes()
    manifest_hash = hashlib.sha256(frozen).hexdigest()
    inputs = json.loads((run_dir / "calibration/inputs.json").read_text())
    if inputs.get("samples_sha256") != manifest_hash:
        raise ValueError("Frozen calibration sample manifest checksum changed")
    samples = [json.loads(line) for line in frozen.decode().splitlines() if line.strip()]
    indices = tuple(indices)
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(samples) for i in indices):
        raise ValueError("Invalid recovery sample indices")
    audit = Path(audit_path) if audit_path is not None else run_dir / "sencache_recovery/assets.json"
    if audit.suffix != ".json": audit = audit / "assets.json"
    if not audit.is_absolute(): audit = run_dir / audit
    state = {"schema_version": 1, "status": "preparing", "manifest_sha256": manifest_hash,
             "indices": list(indices), "download_policy": "serial original HTTPS URLs; Retry-After respected; no source replacement",
             "max_attempts": max_attempts, "records": {}, "events": [],
             "attempts_by_index": {}, "retry_not_before": {}}
    if audit.is_file():
        previous = json.loads(audit.read_text())
        if previous.get("manifest_sha256") != manifest_hash or previous.get("indices") != list(indices):
            raise ValueError("Asset audit belongs to different frozen calibration inputs")
        if previous.get("status") == "passed":
            # Workers pin the complete audit's SHA. A scheduler retry may call
            # preparation again while those workers run; validation must never
            # rewrite successful provenance or downgrade it on an input error.
            records = previous.get("records")
            if not isinstance(records, dict) or set(records) != {str(i) for i in indices}:
                raise ValueError("Passed asset audit has incomplete source records")
            for index in indices:
                sample = samples[index]; path = _video_path(run_dir, sample)
                expected = sample.get("video_sha256")
                if expected is not None and not (isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)):
                    raise ValueError("Expected video checksum must be a lowercase SHA-256")
                url = sample.get("video_url"); parsed = urlsplit(url) if isinstance(url, str) else None
                if not parsed or parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                    raise ValueError("Original video_url must be an explicit HTTPS URL")
                actual = _valid_video_file(path, expected, source_url=url, status="existing")
                saved = records[str(index)]
                if not isinstance(saved, dict) or saved.get("sample_id") != sample["prompt_id"]:
                    raise ValueError("Passed asset audit sample identity differs")
                for key in ("path", "source_url", "bytes", "sha256", "expected_sha256", "checksum_status"):
                    if saved.get(key) != actual[key]:
                        raise ValueError(f"Previously prepared source video or audit changed: sample {index}, {key}")
            if source.read_bytes() != frozen:
                raise ValueError("Calibration manifest changed during asset verification")
            return previous
        for key in ("records", "events", "attempts_by_index", "retry_not_before"):
            if key in previous: state[key] = previous[key]
    def emit(event):
        event = {"time_utc": dt.datetime.fromtimestamp(now(), dt.timezone.utc).isoformat(), **event}
        state.update({key: value for key, value in event.items() if key != "time_utc"})
        state["updated_at"] = event["time_utc"]
        state["events"].append(event)
        atomic_json(audit, state)
        if progress_callback: progress_callback(dict(event))
    def wait_until(index, attempt, deadline):
        remaining = max(0.0, deadline - now())
        while remaining > 0:
            chunk = min(30.0, remaining); sleep(chunk); remaining -= chunk
            emit({"status": "waiting_retry", "index": index, "attempt": attempt,
                  "retry_at": deadline, "wait_seconds_remaining": remaining})
    try:
        for index in indices:
            if source.read_bytes() != frozen: raise ValueError("Calibration manifest changed during asset preparation")
            sample = samples[index]; path = _video_path(run_dir, sample)
            expected = sample.get("video_sha256")
            if expected is not None and not (isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError("Expected video checksum must be a lowercase SHA-256")
            url = sample.get("video_url"); parsed = urlsplit(url) if isinstance(url, str) else None
            if not parsed or parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("Original video_url must be an explicit HTTPS URL")
            if path.exists():
                record = _valid_video_file(path, expected, source_url=url, status="existing")
                observed = state["records"].get(str(index), {}).get("sha256")
                if observed is not None and record["sha256"] != observed:
                    raise ValueError("Previously prepared source video changed; preserved it for inspection")
            else:
                record = None
                first_attempt = state["attempts_by_index"].get(str(index), 0) + 1
                if first_attempt > max_attempts:
                    raise RuntimeError("Asset retry limit exhausted; inspect the saved audit before a new recovery attempt")
                deadline = state["retry_not_before"].get(str(index), 0)
                if deadline > now():
                    emit({"status": "waiting_retry", "index": index, "attempt": first_attempt - 1,
                          "retry_at": deadline, "wait_seconds_remaining": deadline - now()})
                    wait_until(index, first_attempt - 1, deadline)
                for attempt in range(first_attempt, max_attempts + 1):
                    state["attempts_by_index"][str(index)] = attempt
                    emit({"status": "downloading", "index": index, "sample_id": sample["prompt_id"], "attempt": attempt, "retry_at": None})
                    try:
                        record = _download_once(url, path, expected, opener)
                        break
                    except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
                        code = error.code if isinstance(error, HTTPError) else None
                        if (code is not None and code not in RETRYABLE_HTTP) or attempt == max_attempts:
                            raise
                        delay = retry_delay(attempt, getattr(error, "headers", None), now())
                        retry_at = now() + delay
                        state["retry_not_before"][str(index)] = retry_at
                        emit({"status": "waiting_retry", "index": index, "attempt": attempt,
                              "http_status": code, "error": str(error), "retry_at": retry_at,
                              "wait_seconds": delay})
                        # Chunk waits permit the coordinator to report progress/heartbeats.
                        wait_until(index, attempt, retry_at)
                if record is None: raise RuntimeError("Asset preparation ended without a verified source")
            state["records"][str(index)] = {"sample_id": sample["prompt_id"], **record}
            state["retry_not_before"].pop(str(index), None)
            emit({"status": "asset_ready", "index": index, "attempt": 0, "retry_at": None})
        if source.read_bytes() != frozen: raise ValueError("Calibration manifest changed during asset preparation")
        emit({"status": "passed", "index": None, "retry_at": None})
        return state
    except BaseException as error:
        emit({"status": "failed", "error": repr(error), "retry_at": None})
        raise
