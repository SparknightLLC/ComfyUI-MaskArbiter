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
		    },
		    "optional": {
		        "reverse": ("BOOLEAN", {
		            "default": False
		        })
		    }
		}

	CATEGORY = "mask_arbiter"
	FUNCTION = "op"
	RETURN_TYPES = ("MASK", "MASKS")

	def op(self, masks, sort_by, index, reverse):
		masks = list(masks)

		for i in range(len(masks)):
			mask = np.array(masks[i])
			mask = mask.squeeze(0)
			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
			masks[i] = mask

		mask_height, mask_width = masks[0].shape[:2]

		if sort_by == "random":
			import random
			random.shuffle(masks)
		elif sort_by == "largest":
			masks = sorted(masks, key=lambda x: np.sum(x > 0), reverse=False)
		elif sort_by in ("leftmost", "topmost"):
			i = 0
			if sort_by == "topmost":
				i = 1
			bounding_boxes = []
			for c in masks:
				if len(c.shape) == 2 or c.shape[-1] == 1:
					c = cv2.merge([c] * 3)
				gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
				bounding_boxes.append(cv2.boundingRect(cv2.convertScaleAbs(gray)))
			masks = [m for m, _ in sorted(zip(masks, bounding_boxes), key=lambda b: b[1][i])]
		elif sort_by == "innermost":
			center = (mask_width // 2, mask_height // 2)
			masks = sorted(masks, key=lambda x: abs(center[0] - x.shape[0] // 2) + abs(center[1] - x.shape[1] // 2))
		elif sort_by == "merged":
			merged_mask = np.zeros_like(masks[0])
			for mask in masks:
				merged_mask = cv2.add(merged_mask, mask)
			masks = [merged_mask]

		# Ensure that the masks are not a tuple
		masks = list(masks)

		if reverse:
			masks = masks[::-1]

		# Convert mask to expected tensor format
		for i in range(len(masks)):
			mask = cv2.cvtColor(masks[i], cv2.COLOR_BGR2GRAY)
			mask = np.expand_dims(mask, axis=0)
			masks[i] = mask
		masks = torch.tensor(masks)

		return (masks[min(index, len(masks) - 1)], masks)
