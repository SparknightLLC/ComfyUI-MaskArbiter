import torch
import numpy as np
from PIL import Image
import comfy.model_management


def sam_segment(predictor, image, boxes):
	if boxes.shape[0] == 0:
		return None
	image_np = np.array(image)
	image_np_rgb = image_np[..., :3]
	predictor.set_image(image_np_rgb)
	masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=False)
	if masks.ndim == 3:
		masks = np.expand_dims(masks, axis=0)
	return create_tensor_output(image_np, masks)


def groundingdino_predict(dino_model, image, prompt, threshold):
	from .local_groundingdino.datasets import transforms as T

	def load_dino_image(image_pil):
		transform = T.Compose([
		    T.RandomResize([800], max_size=1333),
		    T.ToTensor(),
		    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		])
		image, _ = transform(image_pil, None)  # 3, h, w
		return image

	def get_grounding_output(model, image, caption, box_threshold):
		caption = caption.lower()
		caption = caption.strip()
		if not caption.endswith("."):
			caption = caption + "."
		device = comfy.model_management.get_torch_device()
		image = image.to(device)
		with torch.no_grad():
			outputs = model(image[None], captions=[caption])
		logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
		boxes = outputs["pred_boxes"][0]  # (nq, 4)
		filt_mask = logits.max(dim=1)[0] > box_threshold
		return boxes[filt_mask].cpu()

	dino_image = load_dino_image(image.convert("RGB"))
	boxes_filt = get_grounding_output(dino_model, dino_image, prompt, threshold)
	height, width = image.size[1], image.size[0]
	boxes_filt *= torch.tensor([width, height, width, height], dtype=boxes_filt.dtype)
	boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
	boxes_filt[:, 2:] += boxes_filt[:, :2]
	return boxes_filt


def split_image_mask(image):
	image_rgb = image.convert("RGB")
	image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
	image_rgb = torch.from_numpy(image_rgb)[
	    None,
	]
	if "A" in image.getbands():
		mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
		mask = torch.from_numpy(mask)[
		    None,
		]
	else:
		mask = torch.zeros((1, image.height, image.width), dtype=torch.float32, device="cpu")
	return (image_rgb, mask)


def create_tensor_output(image_np, masks):
	output_masks, output_images = [], []
	for mask in masks:
		mask_pixels = np.any(mask, axis=0)
		image_np_copy = image_np.copy()
		image_np_copy[~mask_pixels] = 0
		output_image, output_mask = split_image_mask(Image.fromarray(image_np_copy))
		output_masks.append(output_mask)
		output_images.append(output_image)
	return (output_images, output_masks)


class GroundingDinoSAM2SegmentList:

	@classmethod
	def INPUT_TYPES(cls):
		return {
		    "required": {
		        "sam_model": ('SAM2_MODEL', {}),
		        "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
		        "image": ('IMAGE', {}),
		        "prompt": ("STRING", {}),
		        "threshold": ("FLOAT", {
		            "default": 0.3,
		            "min": 0,
		            "max": 1.0,
		            "step": 0.01
		        }),
		    }
		}

	CATEGORY = "segment_anything2"
	FUNCTION = "op"
	RETURN_TYPES = ("IMAGE", "MASKS")

	def op(self, grounding_dino_model, sam_model, image, prompt, threshold):
		from .sam2.sam2_image_predictor import SAM2ImagePredictor

		res_images = []
		res_masks = []
		predictor = SAM2ImagePredictor(sam_model)
		for item in image:
			item = Image.fromarray(np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGBA")
			boxes = groundingdino_predict(grounding_dino_model, item, prompt, threshold)
			if boxes.shape[0] == 0:
				continue
			(images, masks) = sam_segment(predictor, item, boxes)
			res_images.extend(images)
			res_masks.extend(masks)
		if len(res_images) == 0:
			_, height, width, _ = image.size()
			empty_image = torch.zeros((1, height, width, 3), dtype=image.dtype, device="cpu")
			empty_mask = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
			return (empty_image, [empty_mask])
		return (torch.cat(res_images, dim=0), res_masks)


NODE_CLASS_MAPPINGS = {
    "GroundingDinoSAM2SegmentList": GroundingDinoSAM2SegmentList,
}

NODE_DISPLAY_NAME_MAPPINGS = {"GroundingDinoSAM2SegmentList": "GroundingDinoSAM2SegmentList"}
