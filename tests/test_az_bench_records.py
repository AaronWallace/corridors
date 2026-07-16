"""Tests for the persistent AlphaZero benchmark record store and fingerprint."""

import socket

from corridors import settings
from corridors.nn import az_selfplay
from corridors.nn.az_bench_store import (
    MAX_RECORDS_PER_FINGERPRINT,
    append_record,
    hardware_fingerprint,
    latest_record,
    load_store,
    records_for,
    select_best,
)


def _use_tmp_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_PATH", tmp_path / "corridors.json")


def _row(positions_per_s, *, fp16=False, **overrides):
    row = {
        "workers": 14, "batch": 256, "concurrency": 8, "servers": 3,
        "fp16": fp16, "batch_timeout_ms": 5.0,
        "evals_per_s": positions_per_s * 40, "positions_per_s": positions_per_s,
        "games_per_s": positions_per_s / 60, "avg_batch": 200.0,
        "fill_pct": 80.0, "request_wait_ms": 4.0, "gpu_busy_pct": 70.0,
    }
    row.update(overrides)
    return row


def test_hardware_key_is_stable_for_identical_inputs():
    make = lambda: az_selfplay.hardware_tuning_key(
        "cuda", 16, "NVIDIA RTX 4090", 24.0, 1, ram_gb=64)
    assert make() == make()
    assert make() != az_selfplay.hardware_tuning_key(
        "cuda", 16, "NVIDIA RTX 4090", 24.0, 1, ram_gb=128)


def test_hardware_key_excludes_transient_identity():
    key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1,
                                          ram_gb=32)
    assert socket.gethostname() not in key
    assert "avail" not in key
    # ram_gb=0 (unknown) keeps the pre-RAM key format for backward compat.
    assert az_selfplay.hardware_tuning_key("cpu", 64) == "cpu|cpu=64"


def test_fingerprint_keeps_only_stable_fields():
    fp = hardware_fingerprint({
        "device": "cuda", "gpu_name": "Example GPU", "vram_gb": 23.988,
        "gpu_count": 1, "ncpu": 16, "ram_gb": 64,
        "avail_gb": 3.2, "hostname": "pod-x7f2", "workers": 14,
    })
    assert fp == {
        "device": "cuda", "gpu_name": "Example GPU", "vram_gb": 24.0,
        "gpu_count": 1, "ncpu": 16, "ram_gb": 64,
    }


def test_legacy_profile_without_ram_segment_still_loads(tmp_path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch)
    legacy = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1)
    az_selfplay.save_tuning_profile(legacy, {"workers": 9})

    keyed = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1,
                                            ram_gb=32)
    assert az_selfplay._load_tuning_profile(keyed) == {"workers": 9}


def test_profile_falls_back_to_committed_benchmark_history(tmp_path, monkeypatch):
    """A fresh machine with no local settings inherits the recorded winner
    for its fingerprint from the (repo-committed) benchmark store."""
    _use_tmp_settings(tmp_path, monkeypatch)
    key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1,
                                          ram_gb=32)
    best = {"workers": 52, "inference_batch": 256, "concurrency": 32,
            "inference_servers": 8, "batch_timeout_ms": 2.0}
    append_record(key, {"rows": [], "best": best})

    assert az_selfplay._load_tuning_profile(key) == best
    # Local settings, once present, still take precedence over history.
    az_selfplay.save_tuning_profile(key, {"workers": 9})
    assert az_selfplay._load_tuning_profile(key) == {"workers": 9}


def test_records_append_retrieve_and_cap(tmp_path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch)
    key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1,
                                          ram_gb=32)
    other = az_selfplay.hardware_tuning_key("cpu", 64, ram_gb=256)
    append_record(other, {"marker": "other"})
    total = MAX_RECORDS_PER_FINGERPRINT + 5
    for index in range(total):
        append_record(key, {"marker": index})

    history = records_for(key)
    assert len(history) == MAX_RECORDS_PER_FINGERPRINT
    assert history[0]["marker"] == total - MAX_RECORDS_PER_FINGERPRINT
    assert latest_record(key)["marker"] == total - 1
    # Other fingerprints are unaffected and live in their own files, so
    # concurrent pods of different hardware classes never merge-conflict.
    assert records_for(other) == [{"marker": "other"}]
    store_files = list((tmp_path / "az_benchmarks").glob("*.json"))
    assert len(store_files) == 2
    assert set(load_store()) == {key, other}


def test_legacy_monolith_is_read_and_superseded_per_key(tmp_path, monkeypatch):
    import json
    _use_tmp_settings(tmp_path, monkeypatch)
    key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1,
                                          ram_gb=32)
    other = az_selfplay.hardware_tuning_key("cpu", 64, ram_gb=256)
    (tmp_path / "az_benchmarks.json").write_text(json.dumps({
        "version": 1,
        "records": {key: [{"marker": "legacy"}], other: [{"marker": "keep"}]},
    }), encoding="utf-8")

    # Legacy records are visible before any new-format write.
    assert records_for(key) == [{"marker": "legacy"}]
    # Appending migrates that key's history (legacy + new) into its own file;
    # untouched keys keep serving from the legacy monolith.
    append_record(key, {"marker": "new"})
    assert records_for(key) == [{"marker": "legacy"}, {"marker": "new"}]
    assert records_for(other) == [{"marker": "keep"}]
    assert len(list((tmp_path / "az_benchmarks").glob("*.json"))) == 1


def test_select_best_prefers_fastest_fp32_and_flags_fp16():
    rows = [_row(100.0), _row(120.0, batch=512), _row(150.0, fp16=True)]
    best, fp16_recommended = select_best(rows)
    assert best["batch"] == 512 and not best["fp16"]
    assert fp16_recommended is True

    best, fp16_recommended = select_best([_row(100.0), _row(90.0, fp16=True)])
    assert best["positions_per_s"] == 100.0
    assert fp16_recommended is False


def test_apply_tuning_profile_applies_new_fields_but_never_fp16(
        tmp_path, monkeypatch):
    _use_tmp_settings(tmp_path, monkeypatch)
    key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1,
                                          ram_gb=32)
    az_selfplay.save_tuning_profile(key, {
        "workers": 12, "inference_batch": 96, "concurrency": 8,
        "inference_servers": 3, "batch_timeout_ms": 7.5,
        "inference_fp16": True,
    })
    result = {
        "hardware_key": key, "workers": 14, "inference_batch": 64,
        "concurrency": 10, "inference_servers": 1, "batch_timeout_ms": 0.0,
        "games_per_iter": 140, "benchmark_tuned": False,
    }

    tuned = az_selfplay._apply_tuning_profile(result)
    assert tuned["inference_servers"] == 3
    assert tuned["batch_timeout_ms"] == 7.5
    assert tuned["workers"] == 12
    assert "inference_fp16" not in tuned
    assert tuned["benchmark_tuned"] is True


def test_record_benchmark_stores_history_and_fp32_profile(tmp_path, monkeypatch):
    from corridors.nn import az_benchmark

    _use_tmp_settings(tmp_path, monkeypatch)
    hw = {"ncpu": 16, "gpu_name": "Example GPU", "vram_gb": 8.0,
          "gpu_count": 1, "ram_gb": 32}

    def metrics(positions_per_s, fp16):
        return {
            "device": "cuda", "workers": 14, "batch_size": 256,
            "concurrency": 8, "inference_servers": 3, "fp16": fp16,
            "batch_timeout_ms": 5.0, "elapsed_s": 10.0,
            "games": 32, "positions": positions_per_s * 10,
            "eval_requests": 4000, "avg_request_wait_ms": 4.0,
            "games_per_s": 3.2, "positions_per_s": positions_per_s,
            "evals_per_s": 400.0,
            "inference": {"batches": 100, "full_batches": 80,
                          "avg_batch": 200.0, "inference_s": 21.0,
                          "num_servers": 3},
        }

    record = az_benchmark.record_benchmark(
        hw, device="cuda", simulations=50, max_plies=12, games=32,
        checkpoint="ck", results=[metrics(500.0, False), metrics(600.0, True)])

    key = az_selfplay.hardware_tuning_key("cuda", 16, "Example GPU", 8.0, 1,
                                          ram_gb=32)
    assert record["fp16_recommended"] is True
    assert record["params"] == {"simulations": 50, "max_plies": 12,
                                "games": 32, "checkpoint": "ck"}
    assert len(record["rows"]) == 2
    assert latest_record(key) == record

    profile = settings.load()["az_tuning_profiles"][key]
    assert profile == {"workers": 14, "inference_batch": 256, "concurrency": 8,
                       "inference_servers": 3, "batch_timeout_ms": 5.0}
    assert "inference_fp16" not in profile
