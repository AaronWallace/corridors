"""Self-play dataset naming tests."""

from corridors.nn.az_menu import _auto_dataset_name


def test_cpu_dataset_name_records_generation_settings():
    name = _auto_dataset_name(
        prefix="az", games=62, simulations=50, max_plies=30,
        device="cpu", workers=31, batch_size=64, concurrency=1,
        search_params={}, timestamp="20260712-120000",
    )
    assert name == (
        "az_20260712-120000_g62_s50_p30_cpu_w31_"
        "cp1p5_da0p3_df0p25_tm20_th1_tl0p1"
    )


def test_gpu_dataset_name_records_batching_and_custom_exploration():
    name = _auto_dataset_name(
        prefix="azloop", games=140, simulations=200, max_plies=150,
        device="cuda", workers=14, batch_size=64, concurrency=10,
        search_params={"c_puct": 2.0, "dirichlet_alpha": 0.05},
        timestamp="20260712-120000",
    )
    assert "_cuda_w14_b64_c10_" in name
    assert "_cp2_da0p05_" in name
