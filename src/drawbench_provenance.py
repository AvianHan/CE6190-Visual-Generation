"""Immutable invocation records and non-destructive resume checks for DrawBench."""
from __future__ import annotations

import hashlib
import json
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_invocation(output_root, environment):
    """Create a new record; never replace the historical env_shard*.json files."""
    root = Path(output_root)
    invocation_id = uuid.uuid4().hex
    relative = Path("invocations") / f"{invocation_id}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(environment, invocation_id=invocation_id, schema_version=SCHEMA_VERSION,
                  created_at=datetime.now(timezone.utc).isoformat())
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"invocation_id": invocation_id, "invocation_file": relative.as_posix(),
            "invocation_sha256": sha256_file(path), "schema_version": SCHEMA_VERSION}


def _png_sha256(path, scientific_config):
    """Check the saved PNG envelope and dimensions, then fingerprint all its bytes."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(33)
        if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[8:16] != b"\x00\x00\x00\rIHDR"):
            raise ValueError(f"Invalid PNG header: {path}")
        width, height = struct.unpack(">II", header[16:24])
        if (width, height) != (scientific_config["width"], scientific_config["height"]):
            raise ValueError(f"PNG dimensions do not match the scientific configuration: {path}")
        stream.seek(-12, 2)
        if stream.read() != b"\x00\x00\x00\x00IEND\xaeB`\x82":
            raise ValueError(f"Incomplete PNG: {path}")
    return sha256_file(path)


def write_result_metadata(stem, record, scientific_config, invocation):
    """Record provenance after a new PNG is saved, without replacing old metadata."""
    stem = Path(stem)
    provenance = dict(invocation, png_sha256=_png_sha256(stem.with_suffix(".png"), scientific_config))
    result = dict(record, scientific_config=scientific_config, provenance=provenance)
    with stem.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write("\n")


def check_existing_result(stem, *, output_root, scientific_config, sample, resume, gpu):
    """Return whether a complete, compatible result can be skipped; fail closed otherwise."""
    stem = Path(stem)
    png, metadata = stem.with_suffix(".png"), stem.with_suffix(".json")
    if not png.exists() and not metadata.exists():
        return False
    message = f"Existing result {stem}: keep it unchanged and use a new output directory."
    if not resume:
        raise ValueError(f"{message} Use --resume only for compatible, complete results.")
    if not png.is_file() or not metadata.is_file():
        raise ValueError(f"{message} Both PNG and JSON are required for resume.")
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if "scientific_config" not in record or "provenance" not in record:
            raise ValueError("Legacy metadata cannot verify the original scientific configuration")
        if not scientific_config.get("revision"):
            raise ValueError("An explicit model --revision is required to verify resumed results")
        if record["scientific_config"] != scientific_config:
            raise ValueError("Scientific configuration differs")
        for key in ("prompt_id", "prompt", "category"):
            if record.get(key) != sample[key]:
                raise ValueError(f"Sample {key} differs")
        for key in ("method", "threshold", "seed"):
            if record.get(key) != scientific_config[key]:
                raise ValueError(f"Sample {key} differs from its configuration")
        if record.get("config_id") != stem.parent.name or record.get("gpu") != gpu:
            raise ValueError("Configuration ID or GPU type differs")
        provenance = record["provenance"]
        if provenance["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported provenance schema")
        root = Path(output_root).resolve()
        relative = Path(provenance["invocation_file"])
        invocation_path = (root / relative).resolve()
        if relative.is_absolute() or not invocation_path.is_relative_to(root / "invocations"):
            raise ValueError("Invocation reference is outside this output directory")
        if sha256_file(invocation_path) != provenance["invocation_sha256"]:
            raise ValueError("Invocation record checksum differs")
        invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
        if (invocation.get("invocation_id") != provenance["invocation_id"]
                or invocation.get("scientific_configs", {}).get(stem.parent.name) != scientific_config):
            raise ValueError("Invocation does not describe this result")
        if _png_sha256(png, scientific_config) != provenance["png_sha256"]:
            raise ValueError("PNG checksum differs")
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ValueError(f"{message} {error}.") from error
    return True
