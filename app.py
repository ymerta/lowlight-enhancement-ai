
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import tempfile
from models.model import UNet

import sys
import os
sys.path.append(os.path.abspath("EnlightenGAN-inference"))
from enlighten_inference import EnlightenOnnxModel
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_type):
    if model_type == "U-Net (MSE)":
        model = UNet().to(device)
        model.load_state_dict(torch.load("models/unet_model_mse.pth", map_location=device))
        model.eval()
        return model
    elif model_type == "U-Net (VGG+SSIM)":
        model = UNet().to(device)
        model.load_state_dict(torch.load("models/unet_vgg_model.pth", map_location=device))
        model.eval()
        return model
    elif model_type == "EnlightenGAN":
        return EnlightenOnnxModel("EnlightenGAN-inference/enlighten_inference/enlighten.onnx", providers=["CPUExecutionProvider"])
    else:
        raise ValueError("Geçersiz model seçimi.")


def enhance_image(image, model_type):
    image_np = np.array(image.convert("RGB"))
    model = load_model(model_type)

    if model_type == "EnlightenGAN":
        output = model.predict(image_np)
    else:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Boyut sabitle
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (out.clip(0, 1) * 255).astype(np.uint8)

    return Image.fromarray(output)


def enhance_video(video_file, model_type):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    model = load_model(model_type)

    # Ortak transform
    to_tensor = transforms.ToTensor()

    def predict_unet(model, img_rgb):
        img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            output = torch.clamp(output, 0, 1)
        output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        return (output * 255).astype(np.uint8)

    def predict_enlighten(model, img_rgb):
        return model.predict(img_rgb)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if model_type == "U-Net (MSE)" or model_type == "U-Net (VGG+SSIM)":
            result = predict_unet(model, img_rgb)
        elif model_type == "EnlightenGAN":
            result = predict_enlighten(model, img_rgb)
        else:
            result = frame  # fallback

        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)

    cap.release()
    out.release()
    return out_path


with gr.Blocks() as demo:
    gr.Markdown("# 🌙 Low-Light Enhancer")
    model_select = gr.Radio(["U-Net (MSE)", "U-Net (VGG+SSIM)", "EnlightenGAN"],
                            value="U-Net (MSE)",
                            label="Model Seç")

    with gr.Tab("📷 Görsel"):
        image_input = gr.Image(type="pil", label="Low-Light Görsel", height=300)
        image_output = gr.Image(type="pil", label="Enhance Sonuç", height=300)
        btn1 = gr.Button("Görseli Enhance Et")
        btn1.click(enhance_image, inputs=[image_input, model_select], outputs=image_output)

    with gr.Tab("🎬 Video"):
        video_input = gr.Video(label="Video Yükle", height=300)
        video_output = gr.Video(label="Enhance Edilmiş Video", height=300)
        btn2 = gr.Button("Videoyu Enhance Et")
        btn2.click(enhance_video, inputs=[video_input, model_select], outputs=video_output)

demo.launch()