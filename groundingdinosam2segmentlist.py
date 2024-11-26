import copy
import torch
import numpy as np
from PIL import Image
import comfy.model_management
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model


def sam_segment(sam_model, image, boxes):
	if boxes.shape[0] == 0:
		return None
	predictor = SAM2ImagePredictor(sam_model)
	image_np = np.array(image)
	image_np_rgb = image_np[..., :3]
	predictor.set_image(image_np_rgb)
	# transformed_boxes = predictor.transform.apply_boxes_torch(
	#     boxes, image_np.shape[:2])
	sam_device = comfy.model_management.get_torch_device()
	masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=False)
	# print("scores: ", scores)

	# prevent merging of masks below
	# masks = np.transpose(masks, (1, 0, 2, 3))
	return create_tensor_output(image_np, masks, boxes)


def groundingdino_predict(dino_model, image, prompt, threshold):

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
		# filter output
		logits_filt = logits.clone()
		boxes_filt = boxes.clone()
		filt_mask = logits_filt.max(dim=1)[0] > box_threshold
		logits_filt = logits_filt[filt_mask]  # num_filt, 256
		boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
		return boxes_filt.cpu()

	dino_image = load_dino_image(image.convert("RGB"))
	boxes_filt = get_grounding_output(dino_model, dino_image, prompt, threshold)
	H, W = image.size[1], image.size[0]
	for i in range(boxes_filt.size(0)):
		boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
		boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
		boxes_filt[i][2:] += boxes_filt[i][:2]
	return boxes_filt


def split_image_mask(image):
	image_rgb = image.convert("RGB")
	image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
	image_rgb = torch.from_numpy(image_rgb)[
	    None,
	]
	if 'A' in image.getbands():
		mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
		mask = torch.from_numpy(mask)[
		    None,
		]
	else:
		mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
	return (image_rgb, mask)


def create_pil_output(image_np, masks, boxes_filt):
	output_masks, output_images = [], []
	boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
	for mask in masks:
		output_masks.append(Image.fromarray(np.any(mask, axis=0)))
		image_np_copy = copy.deepcopy(image_np)
		image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
		output_images.append(Image.fromarray(image_np_copy))
	return output_images, output_masks


def create_tensor_output(image_np, masks, boxes_filt):
	output_masks, output_images = [], []
	boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
	for mask in masks:
		image_np_copy = copy.deepcopy(image_np)
		image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
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
		res_images = []
		res_masks = []
		for item in image:
			item = Image.fromarray(np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
			boxes = groundingdino_predict(grounding_dino_model, item, prompt, threshold)
			if boxes.shape[0] == 0:
				break
			(images, masks) = sam_segment(sam_model, item, boxes)
			res_images.extend(images)
			res_masks.extend(masks)
		if len(res_images) == 0:
			_, height, width, _ = image.size()
			empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
			return (empty_mask, empty_mask)
		return (torch.cat(res_images, dim=0), res_masks)


NODE_CLASS_MAPPINGS = {
    "GroundingDinoSAM2SegmentList": GroundingDinoSAM2SegmentList,
}

NODE_DISPLAY_NAME_MAPPINGS = {"GroundingDinoSAM2SegmentList": "GroundingDinoSAM2SegmentList"}
