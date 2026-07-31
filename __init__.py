from .mask_arbiter import *
from .groundingdinosam2segmentlist import *

__version__ = "0.3.0"

NODE_CLASS_MAPPINGS = {
    "MaskArbiter": MaskArbiter,
    "GroundingDinoSAM2SegmentList": GroundingDinoSAM2SegmentList,
}

NODE_DISPLAY_NAME_MAPPINGS = {"MaskArbiter": "Mask Arbiter", "GroundingDinoSAM2SegmentList": "GroundingDinoSAM2SegmentList"}
