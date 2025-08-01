import streamlit as st
from PIL import Image
import numpy as np
import cv2
import random

st.set_page_config(page_title="🥪 Sandwich Symmetry Evaluator", layout="centered")

# --- Title and Intro ---
st.title("🥪 Sandwich Symmetry Evaluator")
st.subheader("Because symmetry matters. Even in sandwiches.")
st.write("Upload a sandwich image and we'll judge its symmetry... with sass 😎")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload an image of your sandwich (jpg/png)", type=["jpg", "png", "jpeg"])

def evaluate_symmetry(image_pil):
    # Resize and convert to grayscale
    image_pil = image_pil.resize((400, 400))  # Resize for consistency
    image_gray = image_pil.convert("L")  # Grayscale

    # Convert to numpy array
    img_np = np.array(image_gray)

    # Split into left/right
    h, w = img_np.shape
    mid = w // 2
    left = img_np[:, :mid]
    right = img_np[:, mid:]

    # Mirror right half
    right_flipped = cv2.flip(right, 1)

    # Ensure equal size
    min_w = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]

    # Difference map
    diff = cv2.absdiff(left, right_flipped)
    score = 100 - (np.mean(diff) / 255 * 100)  # Higher score = more symmetrical
    score = round(score, 2)

    # Visuals
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return score, diff, heatmap, binary

def generate_comment(score):
    # Generative quirky responses
    if score > 90:
        return random.choice([
            "That's *chef’s kiss* level symmetry! Even the gods of geometry approve.",
            "Did you use a ruler to make this sandwich?!",
            "This symmetry is more satisfying than a perfectly folded samosa 😌",
        ])
    elif score > 70:
        return random.choice([
            "Not bad! It’s like when you *almost* fold a dosa perfectly.",
            "Slightly lopsided, but still edible without an existential crisis.",
            "Like a Picasso sandwich — art, but not accurate.",
        ])
    elif score > 50:
        return random.choice([
            "Uh oh, someone sneezed while assembling this.",
            "That’s a symmetry crime. The Sandwich Police are on their way 🚔",
            "You folded this like a rejected origami class dropout.",
        ])
    else:
        return random.choice([
            "Are you sure this wasn’t made blindfolded during an earthquake?",
            "This sandwich’s symmetry is... abstract 🥴",
            "The only thing symmetrical here is my disappointment.",
        ])

if uploaded_file is not None:
    # Load and display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Sandwich", use_container_width=True)

    # Evaluate symmetry
    score, diff, heatmap, binary = evaluate_symmetry(image)

    # Show results
    st.markdown("---")
    st.subheader("📊 Symmetry Score: " + str(score) + " / 100")

    st.success(generate_comment(score))