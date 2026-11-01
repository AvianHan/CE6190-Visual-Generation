#!/usr/bin/env python3
"""Recover one missing SenCache shard without editing the shared runtime.

The original calibration module still owns the estimator, resize, normalization
and profile format. Only video frame counting is replaced: count the frames
actually yielded by one decoder pass, reopen the same decoder, then select the
original uniform np.linspace indices. Successful existing shards are never
overwritten. Public video assets must already have been staged by the recovery
controller; this GPU worker performs no network download.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_video_record(sample):
    """The original existing-file record, with no network fallback on a GPU."""
    path = Path(sample["video_path"]).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Recovery asset must be staged before GPU calibration: {path}")
    expected = sample.get("video_sha256")
    actual = file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Calibration video checksum mismatch: {path}")
    return {"path": str(path), "sha256": actual, "expected_sha256": expected,
            "checksum_status": "matches_expected" if expected is not None else "observed_only",
            "source_url": sample.get("video_url"), "status": "existing", "bytes": path.stat().st_size}


def count_actual_frames(reader):
    """Enumerate without retaining decoded frames; metadata is diagnostic only."""
    metadata_count = None
    metadata_error = None
    try:
        value = reader.count_frames()
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            metadata_count = value
        else:
            metadata_error = "count_frames returned a non-positive or non-integer count"
    except Exception as error:
        metadata_error = f"{type(error).__name__}: {error}"
    observed = 0
    for _ in reader:
        observed += 1
    if observed < 1:
        raise ValueError("Video decoder yielded no frames")
    return {"metadata_count_frames": metadata_count,
            "metadata_count_error": metadata_error,
            "actual_decoded_frames": observed}


def iter_selected_frames(reader, total_frames, indices):
    """Recheck the second pass and refuse incomplete or unstable decoder output."""
    wanted = set(indices)
    if type(total_frames) is not int or total_frames < 1:
        raise ValueError("Invalid decoded frame count")
    if not wanted or any(type(index) is not int or not 0 <= index < total_frames for index in wanted):
        raise ValueError("Invalid sampled frame indices")
    observed = 0
    for index, frame in enumerate(reader):
        observed = index + 1
        if observed > total_frames:
            raise ValueError("Decoder yielded more frames on its second pass")
        if index in wanted:
            yield index, frame
    if observed != total_frames:
        raise ValueError(f"Decoder yielded {observed} frames on its second pass; expected {total_frames}")


def decoded_plan(path, num_frames, reader_factory, choose_indices):
    """Pure decoder orchestration; reader lifetime and sampling are explicit."""
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError("Requested frame count must be positive")
    reader = reader_factory(path)
    try:
        details = count_actual_frames(reader)
    finally:
        reader.close()
    indices = choose_indices(details["actual_decoded_frames"], num_frames)
    if len(indices) != num_frames:
        raise ValueError("Frame selector returned the wrong number of indices")
    if any(type(i) is not int or not 0 <= i < details["actual_decoded_frames"] for i in indices):
        raise ValueError("Frame selector returned an invalid index")
    details["selected_frame_indices"] = indices
    return details


def actual_stream_loader(original_loader, decoder_audits):
    """Return a per-process loader; no shared source or module files are changed."""
    def load_video(path, row, device):
        import numpy as np
        import torch
        import torch.nn.functional as F
        import imageio.v2 as iio

        path = Path(path).resolve()
        before_hash = file_hash(path)
        if path.suffix.lower() == ".npy":
            result = original_loader(path, row, device)
            details = {"strategy": "original_npy_mmap", "source_sha256": before_hash,
                       "path": str(path)}
        else:
            factory = lambda source: iio.get_reader(str(source), format="FFMPEG")
            details = decoded_plan(path, row["num_frames"], factory,
                lambda total, wanted: np.linspace(0, total - 1, wanted, dtype=int).tolist())
            indices = details["selected_frame_indices"]
            positions = {}
            for output_index, frame_index in enumerate(indices):
                positions.setdefault(frame_index, []).append(output_index)
            output = torch.empty((row["num_frames"], 3, row["height"], row["width"]),
                                 device=device, dtype=torch.float32)
            reader = factory(path)
            try:
                for frame_index, frame in iter_selected_frames(reader, details["actual_decoded_frames"], indices):
                    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
                        raise ValueError(f"{path}: decoded frame {frame_index} is not uint8 H,W,3 RGB")
                    # Match immutable cachebench.calibrate.load_video exactly.
                    x = torch.from_numpy(frame.copy()).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                    x = F.interpolate(x, size=(row["height"], row["width"]), mode="bilinear",
                                      align_corners=False, antialias=True)
                    x = x / 127.5 - 1
                    for output_index in positions[frame_index]:
                        output[output_index].copy_(x[0])
                result = output.permute(1, 0, 2, 3)
            finally:
                reader.close()
            details.update(strategy="two_pass_actual_imageio_ffmpeg_stream", path=str(path),
                           source_sha256=before_hash,
                           resize="original float32 GPU bilinear; align_corners=False; antialias=True; x/127.5-1")
            for package in ("imageio", "imageio-ffmpeg"):
                try:
                    details[package + "_version"] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    details[package + "_version"] = "unavailable"
        if file_hash(path) != before_hash:
            raise ValueError("Source video changed while decoding")
        decoder_audits.append(details)
        return result
    return load_video


def original_arguments(run_dir, sample_index, output):
    """Validate the immutable selection and return original calibration argv."""
    run_dir = Path(run_dir).expanduser().resolve()
    root = run_dir / "calibration"
    plan = json.loads((run_dir / "plan.json").read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    sample_text = (root / "samples.jsonl").read_text()
    config = json.loads((root / "config.json").read_text())
    if hashlib.sha256(sample_text.encode()).hexdigest() != inputs["samples_sha256"]:
        raise ValueError("Calibration samples changed after submission")
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != inputs["config_sha256"]:
        raise ValueError("Calibration configuration changed after submission")
    samples = [json.loads(line) for line in sample_text.splitlines() if line.strip()]
    if type(sample_index) is not int or not 0 <= sample_index < len(samples):
        raise ValueError("Calibration sample index is out of range")
    output = Path(output).expanduser().resolve()
    expected = root / "sencache" / f"shard-{sample_index:04d}.json"
    if output != expected.resolve():
        raise ValueError("Recovery must publish this sample's original shard path")
    if output.exists():
        raise FileExistsError(f"Successful or existing shard will not be overwritten: {output}")
    sample = samples[sample_index]
    video = Path(sample["video_path"]).expanduser()
    if not video.is_absolute():
        video = root / video
    video = video.resolve()
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError(f"Recovery asset must be downloaded before requesting GPU work: {video}")
    actual_hash = file_hash(video)
    if sample.get("video_sha256") is not None and sample["video_sha256"] != actual_hash:
        raise ValueError("Existing calibration video does not match its pinned checksum")
    assets_path = run_dir / "sencache_recovery/assets.json"
    assets_bytes = assets_path.read_bytes()
    assets = json.loads(assets_bytes)
    if assets.get("status") != "passed" or assets.get("manifest_sha256") != inputs["samples_sha256"]:
        raise ValueError("Recovery asset preparation has not passed for these frozen samples")
    asset = assets.get("records", {}).get(str(sample_index), {})
    if (sample_index not in assets.get("indices", []) or asset.get("sample_id") != sample["prompt_id"]
            or Path(asset.get("path", "")).resolve() != video
            or asset.get("sha256") != actual_hash):
        raise ValueError("Staged calibration video differs from the passed asset audit")
    argv = ["cachebench.calibrate", "--method", "sencache", "--config", str(root / "config.json"),
            "--samples", str(root / "samples.jsonl"), "--vendor-dir", str(run_dir / "vendor"),
            "--checkpoint-dir", plan["checkpoint"], "--sample-index", str(sample_index),
            "--output", str(output)]
    for exclusion in inputs["exclusions"]:
        path = Path(exclusion["path"])
        if file_hash(path) != exclusion["sha256"]:
            raise ValueError(f"Held-out prompt file changed: {path}")
        argv += ["--exclude-prompts", str(path)]
    return argv, {"sample_index": sample_index, "prompt_id": sample["prompt_id"],
                  "source_path": str(video), "source_sha256": actual_hash,
                  "assets_audit_path": str(assets_path),
                  "assets_audit_sha256": hashlib.sha256(assets_bytes).hexdigest(),
                  "samples_sha256": inputs["samples_sha256"], "config_sha256": inputs["config_sha256"]}


def run_shard(args):
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".recovery.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        argv, identity = original_arguments(run_dir, args.sample_index, output)
        if audit_output.exists():
            raise FileExistsError("Choose a new recovery attempt audit path")
        from cachebench import calibrate
        source = Path(calibrate.__file__).resolve()
        if source != (run_dir / "code/src/calibrate.py").resolve():
            raise ValueError("The active environment imports a different calibration runtime")
        original_loader = calibrate.load_video
        original_fetch = calibrate.fetch_calibration_video
        original_writer = calibrate.write_profile_atomic
        original_argv = sys.argv
        audits = []
        identity.update(worker_sha256=file_hash(__file__), original_calibrate_sha256=file_hash(source),
                        policy="preserve source and uniform sampling; count actual decoder stream",
                        is_model_training=False)
        def publish(path, data):
            if Path(path).resolve() != output:
                raise ValueError("Recovery attempted to publish an unexpected profile")
            if len(audits) != 1 or audits[0]["source_sha256"] != identity["source_sha256"]:
                raise ValueError("Missing or mismatched decoder audit")
            if file_hash(identity["assets_audit_path"]) != identity["assets_audit_sha256"]:
                raise ValueError("Passed asset audit changed during GPU calibration")
            recovery = {**identity, "decoder": audits[0]}
            data = {**data, "recovery_decoder": recovery}
            original_writer(audit_output, {"status": "calibration_complete", "output": str(output),
                                         "recovery_decoder": recovery})
            original_writer(path, data)
        try:
            calibrate.load_video = actual_stream_loader(original_loader, audits)
            calibrate.fetch_calibration_video = local_video_record
            calibrate.write_profile_atomic = publish
            sys.argv = argv
            calibrate.main()
        finally:
            calibrate.load_video = original_loader
            calibrate.fetch_calibration_video = original_fetch
            calibrate.write_profile_atomic = original_writer
            sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    run_shard(parser.parse_args())


if __name__ == "__main__":
    main()
