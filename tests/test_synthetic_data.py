import torch

from sdam.data.synthetic_video import SyntheticVideoConfig, SyntheticVideoDataset


def make_dataset(seed: int = 7) -> SyntheticVideoDataset:
    config = SyntheticVideoConfig(
        image_size=32,
        channels=3,
        sequence_length=5,
        dataset_size=8,
        object_size=4,
        clutter_count=2,
        min_speed=1.0,
        max_speed=2.0,
    )
    return SyntheticVideoDataset(config=config, seed=seed)


def test_synthetic_video_sample_shapes_and_dtype():
    sample = make_dataset()[0]

    assert sample["obs"].shape == (5, 3, 32, 32)
    assert sample["obs"].dtype == torch.float32
    assert sample["dynamic_mask"].shape == (5, 1, 32, 32)
    assert sample["background"].shape == (3, 32, 32)
    assert sample["target_position"].shape == (2,)
    assert sample["target_velocity"].shape == (2,)


def test_target_moves_and_mask_is_non_empty():
    sample = make_dataset()[0]

    centers = sample["target_positions"]
    assert centers.shape == (5, 2)
    assert not torch.allclose(centers[0], centers[-1])
    assert sample["dynamic_mask"].sum() > 0


def test_dataset_is_deterministic_by_index_and_seed():
    first = make_dataset(seed=11)[3]
    second = make_dataset(seed=11)[3]

    assert torch.allclose(first["obs"], second["obs"])
    assert torch.allclose(first["target_positions"], second["target_positions"])
