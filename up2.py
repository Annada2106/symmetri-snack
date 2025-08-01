import streamlit as st
from PIL import Image
import numpy as np
import cv2
import google.generativeai as genai
import io
import time
import re  # For parsing bounding box coordinates from AI response

# --- Page Configuration ---
st.set_page_config(page_title="🥪 Sandwich Symmetry Evaluator", layout="centered")

# --- Gemini Setup ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.0-flash') 
    vision_model = genai.GenerativeModel("gemini-2.0-flash") 
except Exception as e:
    st.error(f"Configuration Error: Could not load Gemini API key. Details: {e}")
    st.stop()

# --- UI Styling ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #16222A, #3A6073);
            color: white;
        }
        .stFileUploader, .stButton, .stTextInput {
            background-color: #222;
            border: 1px solid #555;
            color: white;
        }
        .symmetry-box {
            background-color: #1f1f2e;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 0 20px #7b2ff7;
            margin-top: 20px;
        }
        .high-score {
            animation: glow 1s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc; }
            to { text-shadow: 0 0 20px #00ffee, 0 0 30px #00ffee; }
        }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_image_bytes_and_mime(image_pil, uploaded_file_type):
    image_bytes_io = io.BytesIO()
    image_pil.save(image_bytes_io, format=image_pil.format if image_pil.format else 'PNG') 
    return image_bytes_io.getvalue(), uploaded_file_type

def get_sandwich_bounding_box(image_bytes, mime_type, img_width, img_height):
    try:
        prompt = (
            "Identify the main sandwich in this image. "
            "Provide its bounding box coordinates in the format: "
            "x_min, y_min, x_max, y_max (as percentages of image width/height). "
            "If no clear sandwich is present, respond with 'None'."
        )
        response = vision_model.generate_content([
            prompt,
            {'mime_type': mime_type, 'data': image_bytes}
        ])
        
        box_str = response.text.strip()
        match = re.search(r'(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)', box_str)
        if match:
            x_min_perc, y_min_perc, x_max_perc, y_max_perc = map(float, match.groups())
            x_min = int(x_min_perc / 100 * img_width)
            y_min = int(y_min_perc / 100 * img_height)
            x_max = int(x_max_perc / 100 * img_width)
            y_max = int(y_max_perc / 100 * img_height)
            x_min = max(0, x_min - int(img_width * 0.02))
            y_min = max(0, y_min - int(img_height * 0.02))
            x_max = min(img_width, x_max + int(img_width * 0.02))
            y_max = min(img_height, y_max + int(img_height * 0.02))
            return (x_min, y_min, x_max, y_max)
        else:
            st.warning(f"AI could not parse bounding box: {box_str}")
            return None
    except Exception as e:
        st.error(f"Error getting bounding box from AI: {e}")
        return None

def analyze_filling_symmetry(cropped_image_bytes, mime_type):
    try:
        prompt = (
            "This is an image of a sandwich. Focus only on the layers between the bread. "
            "Describe how evenly and symmetrically the filling ingredients (e.g., cheese, meat, vegetables) "
            "are distributed and aligned. Be concise and objective."
        )
        response = vision_model.generate_content([
            prompt,
            {'mime_type': mime_type, 'data': cropped_image_bytes}
        ])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error analyzing filling symmetry: {e}")
        return "AI could not analyze filling symmetry."

def generate_comment(score, ai_sandwich_analysis, filling_analysis_description):
    prompt = f"""
You are a sarcastic and witty food critic who only reviews the symmetry of sandwiches.
A sandwich just scored {score}/100 in a symmetry test.
Overall AI observation: "{ai_sandwich_analysis}"
Detailed filling analysis: "{filling_analysis_description}"

Write a short, quirky, one-line review filled with humor and sass.
"""
    retries = 0
    max_retries = 5 
    base_delay = 1 
    while retries < max_retries:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = base_delay * (2 ** retries)
                st.warning(f"Quota exceeded. Retrying in {delay:.1f}s... (Attempt {retries + 1})")
                time.sleep(delay)
                retries += 1
            else:
                return f"AI comment generation failed: {e}"
    return "AI comment generation failed after multiple retries."

def evaluate_symmetry_and_components(image_pil, uploaded_file_type):
    original_width, original_height = image_pil.size
    image_bytes, mime_type = get_image_bytes_and_mime(image_pil, uploaded_file_type)

    # --- AI Cropping ---
    st.info("Detecting sandwich boundaries with AI...")
    bbox = get_sandwich_bounding_box(image_bytes, mime_type, original_width, original_height)
    
    cropped_image_pil = image_pil
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        cropped_image_pil = image_pil.crop((x_min, y_min, x_max, y_max))
        # Removed cropped image display
    else:
        st.warning("Could not automatically crop to sandwich. Analyzing full image.")

    cropped_image_bytes, _ = get_image_bytes_and_mime(cropped_image_pil, uploaded_file_type)

    # --- Symmetry Calculation ---
    image_pil_resized = cropped_image_pil.resize((400, 400))
    image_gray = image_pil_resized.convert("L")
    img_np = np.array(image_gray)

    h, w = img_np.shape
    mid = w // 2
    left = img_np[:, :mid]
    right = img_np[:, mid:]
    right_flipped = cv2.flip(right, 1)

    min_w = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_w]
    right_flipped = right_flipped[:, :min_w]

    diff = cv2.absdiff(left, right_flipped)
    score = 100 - (np.mean(diff) / 255 * 100)
    score = round(score, 2)

    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # --- AI Sandwich Check ---
    ai_sandwich_analysis = "AI could not provide overall analysis."
    is_actually_sandwich = False
    try:
        vision_response_is_sandwich = vision_model.generate_content([
            "Is this an image of a sandwich? Just reply with 'Yes' or 'No'.",
            {'mime_type': mime_type, 'data': image_bytes}
        ])
        if "yes" in vision_response_is_sandwich.text.strip().lower():
            is_actually_sandwich = True
            vision_response_desc = vision_model.generate_content([
                "Describe the main food item in this image. Be concise and objective.",
                {'mime_type': mime_type, 'data': image_bytes}
            ])
            ai_sandwich_analysis = vision_response_desc.text.strip()
        else:
            ai_sandwich_analysis = f"AI identified: {vision_response_is_sandwich.text.strip()}"
    except Exception as e:
        st.error(f"Error during sandwich check: {e}")
        ai_sandwich_analysis = f"Initial sandwich check failed: {e}"

    # --- AI Filling Analysis ---
    filling_analysis_description = "AI could not analyze filling symmetry."
    if is_actually_sandwich:
        st.info("Analyzing filling symmetry with AI...")
        filling_analysis_description = analyze_filling_symmetry(cropped_image_bytes, mime_type)
    
    return score, diff, heatmap, binary, ai_sandwich_analysis, is_actually_sandwich, filling_analysis_description

# --- Main Interface ---
st.title("🥪 Sandwich Symmetry Evaluator")
st.subheader("Upload your sandwich masterpiece")
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Sandwich", use_container_width=True)

    with st.spinner("Processing your sandwich with advanced AI..."):
        score, diff, heatmap, binary, ai_sandwich_analysis, is_actually_sandwich, filling_analysis_description = \
            evaluate_symmetry_and_components(image, uploaded_file.type)

    if not is_actually_sandwich:
        st.error("🚫 That doesn't look like a sandwich. Please upload something more edible.")
        st.info(f"AI's observation: {ai_sandwich_analysis}")
    else:
        if score >= 80:
            st.balloons()
            st.snow()
            score_class = "high-score"
        else:
            score_class = ""

        st.markdown(f"""
        <div class="symmetry-box {score_class}">
            <h3>📊 Symmetry Score (Cropped): <span style="color:#00FFAD">{score}</span> / 100</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**AI's Overall Sandwich Analysis:** {ai_sandwich_analysis}")
        st.markdown(f"**AI's Filling Symmetry Analysis:** {filling_analysis_description}")

        with st.spinner("Generating brutally honest comment..."):
            comment = generate_comment(score, ai_sandwich_analysis, filling_analysis_description)
            st.success("💬 " + comment)
