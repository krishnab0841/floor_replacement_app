import gradio as gr
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import os

# --- AI Model Setup ---
# Load the pre-trained image segmentation model and its processor from Hugging Face.
# This happens once when the script starts.
# The model is "SegFormer", fine-tuned on the ADE20K dataset which includes a "floor" category.
try:
    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an active internet connection to download the model.")
    # You can add fallback logic here if needed
    image_processor = None
    model = None

def segment_floor(image: Image.Image) -> Image.Image:
    """
    Segments the floor in an image using a pre-trained SegFormer model.

    This function takes a PIL image, processes it, and feeds it to a neural network.
    The network predicts the category for each pixel (e.g., wall, ceiling, floor).
    It then creates a black and white mask where the floor is white.
    """
    if image_processor is None or model is None:
        raise gr.Error("AI model failed to load. Please check your internet connection and restart the script.")

    # Ensure the image is in RGB format, as required by the model
    image = image.convert("RGB")

    # Prepare the image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    # Make predictions with the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.cpu()

    # The model outputs a low-resolution segmentation map. We resize it to match the original image.
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # Resize to (height, width)
        mode="bilinear",
        align_corners=False,
    )

    # Find the predicted class for each pixel.
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # In the ADE20K dataset, the 'floor' class has an index of 3.
    # We create a binary mask: 255 (white) for floor pixels, 0 (black) for everything else.
    floor_mask_np = (pred_seg == 3).numpy().astype(np.uint8) * 255
    
    return Image.fromarray(floor_mask_np)

def replace_floor(office_image_np, carpet_image_np):
    """
    Replaces the floor in the office image with the carpet image using the AI-generated mask.
    """
    if office_image_np is None or carpet_image_np is None:
        return None

    office_image = Image.fromarray(office_image_np).convert("RGBA")
    carpet_image = Image.fromarray(carpet_image_np).convert("RGBA")

    # 1. Get the accurate floor mask from the AI model
    gr.Info("Detecting the floor with AI...")
    floor_mask = segment_floor(office_image).convert("L")

    # 2. Tile the carpet texture to match the office image dimensions
    office_width, office_height = office_image.size
    carpet_width, carpet_height = carpet_image.size
    
    tiled_carpet = Image.new('RGBA', (office_width, office_height))
    for i in range(0, office_width, carpet_width):
        for j in range(0, office_height, carpet_height):
            tiled_carpet.paste(carpet_image, (i, j))

    # 3. Combine the original office image with the new carpet using the mask
    # The mask ensures that the new carpet is only applied to the floor area.
    gr.Info("Applying new floor texture...")
    office_image.paste(tiled_carpet, (0, 0), floor_mask)

    return office_image.convert("RGB")

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Virtual Office Floor Planner (AI Edition)
        Upload an image of an office and a texture for the carpet or tiles. 
        The application will use an AI model to accurately detect the floor and replace it with the new texture.
        """
    )
    
    with gr.Row():
        with gr.Column():
            office_input = gr.Image(label="Office Image", type="numpy")
            carpet_input = gr.Image(label="Carpet/Tile Texture", type="numpy")
            submit_button = gr.Button("Generate New Floor")
        with gr.Column():
            output_image = gr.Image(label="Result", type="pil")

    submit_button.click(
        fn=replace_floor,
        inputs=[office_input, carpet_input],
        outputs=output_image
    )
    
    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=[
            ["examples/office1.jpg", "examples/carpet1.jpg"],
            ["examples/office2.jpg", "examples/carpet2.jpg"],
        ],
        inputs=[office_input, carpet_input]
    )

if __name__ == "__main__":
    # Create dummy example files for Gradio examples to work
    if not os.path.exists("examples"):
        os.makedirs("examples")
    try:
        Image.new('RGB', (800, 600), color = '#D3D3D3').save('examples/office1.jpg')
        Image.new('RGB', (150, 150), color = '#4682B4').save('examples/carpet1.jpg')
        Image.new('RGB', (800, 600), color = '#F5F5DC').save('examples/office2.jpg')
        Image.new('RGB', (150, 150), color = '#8B4513').save('examples/carpet2.jpg')
    except Exception as e:
        print(f"Could not create example images: {e}")

    # The 'share=True' parameter creates a public link for Colab
    demo.launch(debug=True, share=True)

