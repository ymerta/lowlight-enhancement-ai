# lowlight-enhancement-ai
# 🌙 Low-Light Enhancer

Enhance low-light **images** and **videos** using deep learning models like U-Net and EnlightenGAN – all in your browser thanks to Gradio and Hugging Face Spaces.

## 🚀 Try the App
👉 [Launch on Hugging Face Spaces](https://huggingface.co/spaces/yourusername/low-light-enhancer)

## 🔍 Features

- 🖼️ **Image** & 🎬 **Video** enhancement  
- ✨ **3 model options**:  
  - U-Net (MSE)  
  - U-Net (VGG + SSIM)  
  - EnlightenGAN (ONNX)  
- 💡 **Real-time enhancement on the web**  
- 🛠️ **Hugging Face Spaces deployment**, no server required  
 

## 🧠 Models Used

🔹 **U-Net (MSE):**  
Classic encoder-decoder CNN trained with MSE loss to brighten images by minimizing pixel-wise differences.

🔹 **U-Net (VGG + SSIM):**  
Enhanced perceptual quality using a combination of VGG feature loss and Structural Similarity Index (SSIM) to make images look more natural.

🔹 **EnlightenGAN (ONNX):**  
A GAN-based model that doesn't need paired training data. Exported in ONNX format for faster, lightweight inference.
 

## 🛠️ Run Locally

```bash
git clone https://github.com/ymerta/lowlight-enhancement-ai.git
cd low-light-enhancer
pip install -r requirements.txt
python app.py

