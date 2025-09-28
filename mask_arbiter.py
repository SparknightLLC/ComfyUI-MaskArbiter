import torch
import numpy as np
import cv2


class MaskArbiter:

	@classmethod
	def INPUT_TYPES(cls):
		return {
		    "required": {
		        "masks": ("MASKS", {}),
		        "sort_by": (["initial", "random", "merged", "leftmost", "topmost", "innermost", "largest"], {
		            "default": "leftmost"
		        }),
		        "index": ("INT", {
		            "default": 0
		        }),
		        "resolution": ("INT", {
		            "default": 64,
		            "min": 0,
		            "tooltip": "The maximum pixel size of the masks for evaluation. 0 to disable. The computation time increases with mask resolution. Note that the size of the returned mask(s) will be the same as your input mask(s)."
		        }),
		    },
		    "optional": {
		        "average": ("BOOLEAN", {
		            "default": False,
		            "tooltip": "If enabled, the average position of mask pixels will be used for sorting instead of the most extreme position. May increase computation time, but is often useful for 'innermost' with masks that have multiple disconnected regions."
		        }),
		        "reverse": ("BOOLEAN", {
		            "default": False
		        })
		    }
		}

	CATEGORY = "mask_arbiter"
	FUNCTION = "op"
	RETURN_TYPES = ("MASK", "MASKS")

	def op(self, masks, sort_by, index, resolution, average, reverse):
		new_masks = list(masks)

		# Convert masks to correct format
		new_masks_out = []
		for i, m in enumerate(new_masks):
			# 1. Ensure NumPy array, avoid deep copy if possible
			if isinstance(m, torch.Tensor):
				mask = m.detach().cpu().numpy()
			else:
				mask = np.asarray(m)

			# 2. Ensure correct shape
			if mask.ndim == 3 and mask.shape[0] == 1:
				mask = mask.squeeze(0)

			if mask.ndim != 2:
				print(f"Skipping mask {i}, unexpected shape:", mask.shape)
				continue

			# 3. Convert to BGR
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

			new_masks_out.append(mask)
		new_masks = new_masks_out

		# print("Resizing masks if needed...")
		# Resize masks for faster computation, preserving aspect ratio
		if resolution > 1:
			small_masks = []

			height, width = new_masks[0].shape[:2]
			new_width = resolution
			new_height = int((height / width) * resolution)

			for mask in new_masks:
				small_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
				small_masks.append(small_mask)
		else:
			small_masks = new_masks

		mask_height, mask_width = small_masks[0].shape[:2]
		order = None

		# print("Performing mask sorting...")
		if sort_by == "random":
			import random
			order = list(range(len(new_masks)))
			random.shuffle(order)
		elif sort_by == "largest":
			order = sorted(range(len(small_masks)), key=lambda i: np.sum(small_masks[i] > 0))
		elif sort_by in ("leftmost", "topmost"):
			i = 0 if sort_by == "leftmost" else 1
			if average:
				# Compute the average position of non-zero pixels
				order = sorted(range(len(small_masks)), key=lambda idx: np.mean(np.where(small_masks[idx] > 0)[i]) if np.any(small_masks[idx] > 0) else float("inf"))
			else:
				bounding_boxes = [cv2.boundingRect(cv2.convertScaleAbs(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))) for mask in small_masks]
				order = sorted(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i][1] if sort_by == "topmost" else bounding_boxes[i][0])
		elif sort_by == "innermost":
			center = (mask_width // 2, mask_height // 2)
			if average:
				# Compute the average position of non-zero pixels
				order = sorted(range(len(small_masks)), key=lambda i: (np.mean(np.where(small_masks[i] > 0)[1]) - center[0])**2 + (np.mean(np.where(small_masks[i] > 0)[0]) - center[1])**2 if np.any(small_masks[i] > 0) else float("inf"))
			else:
				order = sorted(range(len(small_masks)), key=lambda i: abs(center[0] - small_masks[i].shape[1] // 2) + abs(center[1] - small_masks[i].shape[0] // 2))
		if sort_by == "merged":
			merged_mask = np.sum(masks, axis=0, dtype=np.uint8)
			new_masks = [merged_mask]

		# print("Applying order to the original masks...")
		# Apply order to the original masks
		if order:
			masks = [masks[i] for i in order]

		# print("Reversing mask order if needed...")
		if reverse:
			masks = masks[::-1]

		# Convert masks to the expected tensor format

		# print("Returning mask...")
		return (masks[min(index, len(masks) - 1)], masks)
