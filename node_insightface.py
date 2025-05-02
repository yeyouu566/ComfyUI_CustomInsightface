class InsightFaceLoader:
    def __init__(self):
        self.app = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("insightface",)  # 🔧 여기 이름을 IPAdapter가 요구하는 것과 동일하게!
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
