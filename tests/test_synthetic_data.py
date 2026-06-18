import pytest
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
    assert sample["target_positions"].dtype == torch.float32
    assert sample["target_position"].dtype == torch.float32
    assert sample["target_velocity"].dtype == torch.float32
    assert sample["dynamic_mask"].dtype == torch.float32
    assert sample["background"].dtype == torch.float32


def test_target_moves_and_mask_is_non_empty():
    sample = make_dataset()[0]

    centers = sample["target_positions"]
    assert centers.shape == (5, 2)
    assert not torch.allclose(centers[0], centers[-1])
    assert sample["dynamic_mask"].sum() > 0


def test_target_labels_match_dynamic_mask_centers_and_velocity():
    sample = make_dataset()[0]

    mask_centers = []
    for frame_mask in sample["dynamic_mask"][:, 0]:
        coords_yx = torch.nonzero(frame_mask, as_tuple=False)
        assert coords_yx.numel() > 0
        center = torch.stack(
            [
                coords_yx[:, 1].float().mean(),
                coords_yx[:, 0].float().mean(),
            ]
        )
        mask_centers.append(center)
    mask_centers = torch.stack(mask_centers)

    assert torch.allclose(sample["target_positions"], mask_centers)
    assert torch.allclose(sample["target_position"], mask_centers[-1])
    assert torch.allclose(sample["target_velocity"], mask_centers[-1] - mask_centers[-2])


def test_dataset_is_deterministic_by_index_and_seed():
    first = make_dataset(seed=11)[3]
    second = make_dataset(seed=11)[3]

    assert torch.allclose(first["obs"], second["obs"])
    assert torch.allclose(first["target_positions"], second["target_positions"])


def test_adjacent_seed_and_index_pairs_do_not_collide():
    first = make_dataset(seed=11)[3]
    second = make_dataset(seed=12)[2]

    assert not (
        torch.allclose(first["obs"], second["obs"])
        and torch.allclose(first["target_positions"], second["target_positions"])
    )


def test_dataset_sampling_does_not_advance_global_torch_rng_state():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()

    _ = make_dataset(seed=17)[4]

    after = torch.random.get_rng_state()
    assert torch.equal(after, before)


def test_synthetic_samples_have_visible_motion_labels_across_seed_sweep():
    for seed in [*range(3), 130, 260]:
        dataset = make_dataset(seed=seed)
        for index in range(8):
            sample = dataset[index]
            target_positions = sample["target_positions"]

            assert not torch.allclose(target_positions[-1], target_positions[0])
            assert torch.linalg.vector_norm(sample["target_velocity"]) > 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"object_size": 33}, "object_size must be <= image_size"),
        ({"object_size": 32}, "object_size must be smaller than image_size"),
        ({"clutter_count": -1}, "clutter_count must be >= 0"),
        ({"min_speed": 3.0, "max_speed": 2.0}, "speed range must satisfy 0 < min_speed <= max_speed"),
        ({"min_speed": 0.0, "max_speed": 0.0}, "speed range must satisfy 0 < min_speed <= max_speed"),
        ({"image_size": 32.0}, "image_size must be an int"),
        ({"min_speed": float("nan")}, "min_speed must be finite"),
    ],
)
def test_invalid_synthetic_video_config_values_raise_clear_errors(overrides, message):
    values = {
        "image_size": 32,
        "channels": 3,
        "sequence_length": 5,
        "dataset_size": 8,
        "object_size": 4,
        "clutter_count": 2,
        "min_speed": 1.0,
        "max_speed": 2.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        SyntheticVideoDataset(config=SyntheticVideoConfig(**values), seed=7)
