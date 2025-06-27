# lowlight-enhancement-ai
🌙 Low-Light Image & Video Enhancer

Hugging Face Spaces Demo: 🔗 Live App
Medium Blog: 📝 Medium Post

An AI-powered Gradio application that brightens dark images and videos using deep learning models. Upload your low-light content and enhance it instantly with three model options!

🔍 Features
	•🖼️ Image & 🎬 video enhancement
	•✨ 3 model options: U-Net (MSE), U-Net (VGG + SSIM), EnlightenGAN (ONNX)
	•💡 Real-time enhancement on web
	•🛠️ Hugging Face Spaces deployment, no server required
 

🧠 Models
 U-Net (MSE): Trained with pixel-wise MSE loss. Good brightness boost, may be slightly blurry.
 U-Net (VGG+SSIM): Uses perceptual loss for sharper and more natural results.
 EnlightenGAN: Lightweight GAN in ONNX format. Works well even without paired training data.
 

⚙️ How to Run Locally
git clone https://github.com/ymerta/low-light-enhancer.git
cd low-light-enhancer
pip install -r requirements.txt
python app.py


📁 File Structure
├── models/
│   ├── unet_model_mse.pth
│   ├── unet_vgg_model.pth
│   └── enlighten.onnx
├── EnlightenGAN-inference/
│   └── enlighten_inference.py
├── app.py
├── requirements.txt
