import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mask_arbiter import MaskArbiter


def _mask(height, width, y1, y2, x1, x2):
	mask = torch.zeros((height, width))
	mask[y1:y2, x1:x2] = 1
	return mask


def test_empty_tensor_batch_returns_black_mask():
	empty = torch.zeros((0, 12, 20))
	selected, ordered = MaskArbiter().op("initial", 0, 64, masks=empty)
	assert selected.shape == (1, 12, 20)
	assert not selected.any()
	assert ordered.shape == (0, 12, 20)


def test_positional_and_largest_sorting():
	left = _mask(20, 20, 8, 12, 1, 3)
	right = _mask(20, 20, 2, 12, 12, 19)
	masks = torch.stack((right, left))

	selected, ordered = MaskArbiter().op("leftmost", 0, 0, masks=masks)
	assert torch.equal(selected[0], left)
	assert torch.equal(ordered[0], left)

	selected, _ = MaskArbiter().op("topmost", 0, 0, average=True, masks=masks)
	assert torch.equal(selected[0], right)

	selected, _ = MaskArbiter().op("largest", 0, 0, masks=masks)
	assert torch.equal(selected[0], right)


def test_innermost_uses_mask_position():
	center = _mask(21, 21, 9, 12, 9, 12)
	edge = _mask(21, 21, 0, 4, 0, 4)
	selected, _ = MaskArbiter().op("innermost", 0, 0, masks=torch.stack((edge, center)))
	assert torch.equal(selected[0], center)


def test_merge_and_reverse():
	left = _mask(10, 10, 2, 5, 1, 4)
	right = _mask(10, 10, 2, 5, 6, 9)
	masks = torch.stack((left, right))

	selected, ordered = MaskArbiter().op("merged", 0, 64, masks=masks)
	assert len(ordered) == 1
	assert torch.equal(selected[0], torch.maximum(left, right))

	selected, ordered = MaskArbiter().op("initial", 0, 64, reverse=True, masks=masks)
	assert torch.equal(selected[0], right)
	assert torch.equal(ordered[0], right)
