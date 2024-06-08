import cv2
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import gradio as gr
import sys
sys.path.append("..")
from segment_anything.sam import load
from segment_anything.predictor import SamPredictor

sam_checkpoint = "./sam-vit-base"
sam = load(sam_checkpoint)
predictor = SamPredictor(sam)
from segment_anything.utils.transforms import ResizeLongestSide

resize_transform = ResizeLongestSide(sam.vision_encoder.img_size)

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = mx.array(image)
    return image

def automatic_box_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(approx)
        
        # Filter out small boxes
        if w > 50 and h > 50:
            boxes.append([x, y, x + w, y + h])
    
    return mx.array(boxes)

def process_image(image):
    image_boxes = automatic_box_extraction(image)
    input_data = {
        'image': prepare_image(image, resize_transform, sam),
        'boxes': resize_transform.apply_boxes(image_boxes, image.shape[:2]),
        'original_size': image.shape[:2]
    }
    output_data = sam([input_data], multimask_output=False)[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    for mask in output_data['masks']:
        show_mask(np.array(mask), ax, image, random_color=True, alpha=0.3)
    for box in image_boxes:
        show_box(np.array(box), ax)
    ax.axis('off')
    plt.tight_layout()
    return fig

def show_mask(mask, ax, image, random_color=False, alpha=0.3):
    color = np.random.random(3) if random_color else np.array([0.1, 0.3, 0.8])  # 파란색 계열로 변경
    mask_image = np.dstack([mask]*3) * color
    mask_rgb = np.dstack([mask] * 3)
    mask_rgba = np.dstack([mask_rgb, mask * alpha])
    mask_rgba = cv2.resize(mask_rgba, (image.shape[1], image.shape[0]))
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    image_rgba = cv2.resize(image_rgba, (mask_rgba.shape[1], mask_rgba.shape[0]))
    image_with_mask = cv2.addWeighted(image_rgba, 1, (mask_rgba * 255).astype(np.uint8), alpha, 0)
    ax.imshow(cv2.cvtColor(image_with_mask, cv2.COLOR_RGBA2RGB))

def show_box(box, ax):
    rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

def gradio_interface(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    fig = process_image(image)
    return fig

inputs = gr.Image(type="pil")
outputs = gr.Plot()
gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs, title="Image Segmentation with SAM", description="Upload an image to perform segmentation.").launch()
