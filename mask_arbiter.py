import random

import cv2
import numpy as np
import torch


def _as_mask_list(masks):
	if isinstance(masks, torch.Tensor):
		if masks.ndim == 2:
			return [masks]
		if masks.ndim == 3:
			return list(masks)
		if masks.ndim == 4 and masks.shape[1] == 1:
			return [item[0] for item in masks]
		raise ValueError(f"Expected masks with shape [H,W], [N,H,W], or [N,1,H,W], received {tuple(masks.shape)}.")
	return list(masks)


def _as_evaluation_mask(mask, index):
	if isinstance(mask, torch.Tensor):
		mask = mask.detach().cpu().numpy()
	else:
		mask = np.asarray(mask)

	if mask.ndim == 3 and mask.shape[0] == 1:
		mask = mask[0]
	elif mask.ndim == 3 and mask.shape[-1] == 1:
		mask = mask[..., 0]
	if mask.ndim != 2:
		raise ValueError(f"Mask {index} must be two-dimensional after removing a singleton channel, received {mask.shape}.")
	return (mask > 0).astype(np.uint8, copy=False)


def _resize_for_evaluation(masks, resolution):
	height, width = masks[0].shape
	if any(mask.shape != (height, width) for mask in masks):
		raise ValueError("All masks must have the same dimensions.")
	if resolution <= 0 or max(height, width) <= resolution:
		return masks

	scale = resolution / max(height, width)
	new_width = max(1, round(width * scale))
	new_height = max(1, round(height * scale))
	return [cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST) for mask in masks]


def _merge_masks(masks):
	if all(isinstance(mask, torch.Tensor) for mask in masks):
		normalized = [mask[0] if mask.ndim == 3 and mask.shape[0] == 1 else mask for mask in masks]
		return torch.stack(normalized).max(dim=0).values
	normalized = [_as_evaluation_mask(mask, index) for index, mask in enumerate(masks)]
	return torch.from_numpy(np.maximum.reduce(normalized))


def _as_mask_output(mask):
	if not isinstance(mask, torch.Tensor):
		mask = torch.from_numpy(np.asarray(mask))
	if mask.ndim == 2:
		mask = mask.unsqueeze(0)
	return mask


class MaskArbiter:

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"sort_by": (["initial", "random", "merged", "leftmost", "topmost", "innermost", "largest"], {
					"default": "leftmost",
					"tooltip": "Order masks before selecting an index. 'merged' returns one union of all masks.",
				}),
				"index": ("INT", {
					"default": 0,
					"min": 0,
					"tooltip": "Zero-based index to select after sorting. Values past the end select the last mask.",
				}),
				"resolution": ("INT", {
					"default": 64,
					"min": 0,
					"tooltip": "Maximum width or height used only for sorting. Use 0 for full resolution; outputs retain their original size.",
				}),
			},
			"optional": {
				"masks": ("MASKS", {
					"tooltip": "Mask list or batch to sort. Ignored when 'mask' is connected.",
				}),
				"mask": ("MASK", {
					"tooltip": "Standard ComfyUI mask batch. Takes precedence over 'masks' when both are connected.",
				}),
				"average": ("BOOLEAN", {
					"default": False,
					"tooltip": "Use each mask's centroid for positional sorting. Otherwise, use its nearest relevant edge or pixel.",
				}),
				"reverse": ("BOOLEAN", {
					"default": False,
					"tooltip": "Reverse the final mask order before selecting the index.",
				}),
			},
		}

	CATEGORY = "mask_arbiter"
	DESCRIPTION = "Sort, merge, and select masks while preserving their original resolution."
	FUNCTION = "op"
	RETURN_TYPES = ("MASK", "MASKS")
	RETURN_NAMES = ("selected_mask", "ordered_masks")

	def op(self, sort_by, index, resolution, average=False, reverse=False, masks=None, mask=None):
		source_masks = mask if mask is not None else masks
		if source_masks is None:
			raise ValueError("Connect either 'masks' or 'mask'.")

		mask_list = _as_mask_list(source_masks)
		if len(mask_list) == 0:
			if isinstance(source_masks, torch.Tensor) and source_masks.ndim >= 3:
				empty_mask = torch.zeros((1, source_masks.shape[-2], source_masks.shape[-1]), dtype=source_masks.dtype, device=source_masks.device)
				return (empty_mask, source_masks)
			raise ValueError("The input mask collection is empty and has no dimensions for an empty output mask.")

		if sort_by == "merged":
			mask_list = [_merge_masks(mask_list)]
		elif sort_by in ("initial", "random"):
			if sort_by == "random":
				random.shuffle(mask_list)
		else:
			evaluation_masks = [_as_evaluation_mask(item, item_index) for item_index, item in enumerate(mask_list)]
			evaluation_masks = _resize_for_evaluation(evaluation_masks, resolution)
			positions = [np.nonzero(item) for item in evaluation_masks]

			if sort_by == "largest":
				order = sorted(range(len(mask_list)), key=lambda item_index: -len(positions[item_index][0]))
			elif sort_by in ("leftmost", "topmost"):
				axis = 1 if sort_by == "leftmost" else 0
				def position(item_index):
					coordinates = positions[item_index][axis]
					if len(coordinates) == 0:
						return float("inf")
					return float(coordinates.mean() if average else coordinates.min())
				order = sorted(range(len(mask_list)), key=position)
			elif sort_by == "innermost":
				height, width = evaluation_masks[0].shape
				center_y = (height - 1) / 2
				center_x = (width - 1) / 2
				def center_distance(item_index):
					y_coordinates, x_coordinates = positions[item_index]
					if len(y_coordinates) == 0:
						return float("inf")
					if average:
						return float((y_coordinates.mean() - center_y) ** 2 + (x_coordinates.mean() - center_x) ** 2)
					return float(np.min((y_coordinates - center_y) ** 2 + (x_coordinates - center_x) ** 2))
				order = sorted(range(len(mask_list)), key=center_distance)
			mask_list = [mask_list[item_index] for item_index in order]

		if reverse:
			mask_list.reverse()

		selected_index = min(index, len(mask_list) - 1)
		return (_as_mask_output(mask_list[selected_index]), mask_list)
