import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .mask_arbiter import *
from .groundingdinosam2segmentlist import *

NODE_CLASS_MAPPINGS = {
    "MaskArbiter": MaskArbiter,
    "GroundingDinoSAM2SegmentList": GroundingDinoSAM2SegmentList,
}

NODE_DISPLAY_NAME_MAPPINGS = {"MaskArbiter": "Mask Arbiter", "GroundingDinoSAM2SegmentList": "GroundingDinoSAM2SegmentList"}
