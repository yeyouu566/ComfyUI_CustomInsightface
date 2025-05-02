import os
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

class InsightFaceLoader:
    def __init__(self):
        self.app = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("FACE_INFO",)
    FUNCTION = "analyze"

    CATEGORY = "face"

    def analyze(self, image):
        if self.app is None:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0)

        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = image

        faces = self.app.get(img)
        return (faces,)
