# Floor Replacement App

An AI-powered application that allows users to replace floors in images using semantic segmentation. Built with Python, PyTorch, and Gradio.

## 🚀 Features

- **AI-Powered Floor Segmentation**: Uses SegFormer model to accurately detect and segment floors in images
- **Interactive Interface**: Simple and intuitive Gradio-based web interface
- **Real-time Processing**: See results instantly
- **Customizable**: Easy to modify and extend for different use cases

## 🛠️ Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/krishnab0841/floor_replacement_app.git
   cd floor_replacement_app
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ How to Run

1. Start the application:
   ```bash
   python App.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:7860`)

3. Upload an office/room image and a carpet/flooring image to see the magic happen!

## 🖼️ Example

1. Upload an office/room photo
2. Upload a carpet/flooring pattern
3. Click "Generate" to see the result

## 📦 Dependencies

- gradio
- numpy
- Pillow
- torch
- transformers

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Uses the [SegFormer](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) model from Hugging Face
- Built with [Gradio](https://gradio.app/) for the web interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
