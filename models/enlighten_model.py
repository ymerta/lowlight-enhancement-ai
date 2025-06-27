import onnxruntime as ort
import cv2
import numpy as np

class EnlightenOnnxModel:
    def __init__(self, model_path="models/enlighten.onnx", providers=["CPUExecutionProvider"]):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        img = cv2.resize(image, (512, 512))  # model eğitildiği boyutta resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW
        return img

    def postprocess(self, output):
        out = output.squeeze().transpose(1, 2, 0)  # HWC
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return out

    def predict(self, image):
        input_tensor = self.preprocess(image)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        return self.postprocess(output)