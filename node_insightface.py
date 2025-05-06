TYPE_CLASSES = {
    "INSIGHTFACE": list  # <- 여전히 필요
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

        embeddings = []
        for face in faces:
            if face.embedding is not None:
                embeddings.append(face.embedding.tolist())

        return (embeddings,)  # ✅ 오직 embedding만
