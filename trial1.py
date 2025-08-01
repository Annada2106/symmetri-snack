import streamlit as st
from PIL import Image
import numpy as np

st.title("Sandwich Analyzer")
uploaded_file = st.file_uploader("Upload an image of your sandwich", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Sandwich", use_column_width=True)
import torch
from torchvision import models, transforms

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_prediction(img_pil):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img)
    return prediction

if uploaded_file:
    prediction = get_prediction(img)
    # Display prediction results here
def is_centered(bbox, sandwich_box):
    _, x1, y1, x2, y2 = bbox
    sandwich_center_x = (sandwich_box[1] + sandwich_box[3]) / 2
    ingredient_center_x = (x1 + x2) / 2
    return abs(sandwich_center_x - ingredient_center_x) < symmetry_threshold

# For symmetry, compare left and right side pixels of masks or bounding boxes.
import streamlit as st
from PIL import Image

st.title("Sandwich Analyzer")

uploaded_file = st.file_uploader("Upload an image of your sandwich", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Sandwich", use_column_width=True)
    
    # 1. Identify sandwich type/model prediction
    # 2. Perform ingredient detection (object segmentation)
    # 3. Centering/symmetry scoring logic
    # 4. Overlay detected features, print results
