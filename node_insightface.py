TYPE_CLASSES = {
    "INSIGHTFACE": list
}

class InsightFaceLoader:
    def __init__(self):
        self.app = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INSIGHTFACE",)
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

        face_infos = []
        for face in faces:
            face_infos.append({
                "bbox": face.bbox.tolist() if face.bbox is not None else [],
                "kps": face.kps.tolist() if face.kps is not None else [],
                "gender": float(face.gender) if face.gender is not None else -1,
                "age": float(face.age) if face.age is not None else -1,
                "embedding": face.embedding.tolist() if face.embedding is not None else []
            })

        return (face_infos,)
