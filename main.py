import streamlit as st
from PIL import Image
import numpy as np
import cv2
import google.generativeai as genai
import io
import time
import re
from datetime import datetime
import os

# Import all functions from your database file
from database import (
    setup_database,
    create_user,
    authenticate_user,
    add_history_entry,
    get_user_history
)

# --- Initial Setup ---
setup_database()

# --- Page Configuration ---
st.set_page_config(page_title="Symmetrich", layout="centered")

# --- Gemini Setup ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    vision_model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    st.error(f"Configuration Error: Could not load Gemini API key. Details: {e}")
    st.stop()

# --- Session State Management ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = None
    st.session_state['page'] = 'Login'

# --- UI Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        /* --- Global Styles --- */
        html, body, *, h1, h2, h3, h4, h5, h6, [class*="st-"] {
            font-family: 'Poppins', sans-serif !important;
        }
        .stApp {
            background: linear-gradient(to right, #16222A, #3A6073);
        }

        /* --- Fun Title and Subtitle Styles --- */
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding-top: 1rem;
            padding-bottom: 0.5rem;
            gap: 1.5rem;
        }
        .logo-emoji {
            font-size: 4rem;
            animation: float 4s ease-in-out infinite;
            text-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .logo-emoji:last-child {
            animation-delay: -2s;
        }
        .main-title {
            font-size: 6rem;
            font-weight: 700 !important;
            background: linear-gradient(45deg, #00F5D4, #FF00FF, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #C0C0C0;
            font-weight: 300;
            letter-spacing: 1.5px;
            margin-bottom: 2rem;
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        /* --- Glassmorphism Containers --- */
        .symmetry-box, .history-entry {
            background-color: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 25px;
        }
        .symmetry-box {
            text-align: center;
            box-shadow: 0 0 30px rgba(0, 245, 212, 0.2);
            margin-top: 20px;
        }
        .history-entry {
            margin-bottom: 15px;
        }
        .high-score {
            animation: glow 1.5s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { box-shadow: 0 0 20px rgba(0, 245, 212, 0.3); }
            to { box-shadow: 0 0 40px rgba(0, 245, 212, 0.7); }
        }
        .symmetry-box h3 span {
            color: #00F5D4;
            font-weight: 700 !important;
            font-size: 1.5em;
            text-shadow: 0 0 15px rgba(0, 245, 212, 0.7);
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
        response = vision_model.generate_content([prompt, {'mime_type': mime_type, 'data': image_bytes}])
        box_str = response.text.strip()
        match = re.search(r'(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)', box_str)
        if match:
            x_min_perc, y_min_perc, x_max_perc, y_max_perc = map(float, match.groups())
            x_min = int(x_min_perc / 100 * img_width)
            y_min = int(y_min_perc / 100 * img_height)
            x_max = int(x_max_perc / 100 * img_width)
            y_max = int(y_max_perc / 100 * img_height)

            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(img_width, x_max), min(img_height, y_max)
            padding_x, padding_y = int(img_width * 0.02), int(img_height * 0.02)
            x_min, y_min, x_max, y_max = max(0, x_min - padding_x), max(0, y_min - padding_y), min(img_width, x_max + padding_x), min(img_height, y_max + padding_y)
            
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
        response = vision_model.generate_content([prompt, {'mime_type': mime_type, 'data': cropped_image_bytes}])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error analyzing filling symmetry: {e}")
        return "AI could not analyze filling symmetry."

def evaluate_symmetry_and_components(image_pil, uploaded_file_type):
    original_width, original_height = image_pil.size
    image_bytes, mime_type = get_image_bytes_and_mime(image_pil, uploaded_file_type)
    st.info("Detecting sandwich boundaries with AI...")
    bbox = get_sandwich_bounding_box(image_bytes, mime_type, original_width, original_height)
    
    if bbox and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
        cropped_image_pil = image_pil.crop(bbox)
    else:
        st.warning("Could not automatically crop to sandwich. Analyzing full image.")
        cropped_image_pil = image_pil
    
    cropped_image_bytes, _ = get_image_bytes_and_mime(cropped_image_pil, uploaded_file_type)
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
    score = round(100 - (np.mean(diff) / 255 * 100), 2)
    ai_sandwich_analysis = "AI could not provide overall analysis."
    is_actually_sandwich = False
    try:
        vision_response_is_sandwich = vision_model.generate_content(["Is this an image of a sandwich? Just reply with 'Yes' or 'No'.", {'mime_type': mime_type, 'data': image_bytes}])
        if "yes" in vision_response_is_sandwich.text.strip().lower():
            is_actually_sandwich = True
            vision_response_desc = vision_model.generate_content(["Describe the main food item in this image. Be concise and objective.", {'mime_type': mime_type, 'data': image_bytes}])
            ai_sandwich_analysis = vision_response_desc.text.strip()
        else:
            ai_sandwich_analysis = f"AI identified: {vision_response_is_sandwich.text.strip()}"
            is_actually_sandwich = False
    except Exception as e:
        st.error(f"Error during initial sandwich check: {e}")
        ai_sandwich_analysis = f"Initial sandwich check failed: {e}"
        is_actually_sandwich = False
    filling_analysis_description = "AI could not analyze filling symmetry."
    if is_actually_sandwich:
        st.info("Analyzing filling symmetry with AI...")
        filling_analysis_description = analyze_filling_symmetry(cropped_image_bytes, mime_type)
    return score, diff, None, None, ai_sandwich_analysis, is_actually_sandwich, filling_analysis_description

def generate_comment(score, ai_sandwich_analysis, filling_analysis_description):
    # --- RESTORED a more detailed prompt for better, more accurate critiques ---
    prompt = f"""
You are a sarcastic and witty food critic who only reviews the symmetry of sandwiches.
A sandwich just scored {score}/100 in a symmetry test.
Overall AI observation: "{ai_sandwich_analysis}"
Detailed filling analysis: "{filling_analysis_description}"

Considering the score, the overall observation, AND especially the detailed filling analysis,
write a short, quirky, one-line review filled with humor and sass.
Incorporate specific details from the observations to make the critique more informed and dramatic.
Avoid compliments if the score is low. Be generous if it's very high.
Make it sound like a savage or dramatic food critique.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Comment generation failed due to API error: {e}. Using fallback comment.")
        if score > 85: return f"A respectable {score}/100. Well done."
        else: return f"A score of {score}/100. There's room for improvement."


# --- UI FORMS (Login/Signup) ---
def show_login_page():
    st.header("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = user_id
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    if st.button("Don't have an account? Sign Up"):
        st.session_state['page'] = 'Signup'
        st.rerun()

def show_signup_page():
    st.header("Create Account")
    with st.form("signup_form"):
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            if create_user(username, password):
                st.success("Account created! Please log in.")
                st.session_state['page'] = 'Login'
                st.rerun()
            else:
                st.error("Username already taken. Please choose another.")
    if st.button("Already have an account? Login"):
        st.session_state['page'] = 'Login'
        st.rerun()


# --- MAIN APP DISPLAY ---
def show_main_app():
    st.sidebar.header(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("""
        <div class="title-container">
            <div class="logo-emoji">🥪</div>
            <h1 class="main-title">Symmetrich</h1>
            <div class="logo-emoji">🥪</div>
        </div>
        <p class="subtitle">Upload your sandwich masterpiece</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Sandwich", use_container_width=True)
        with st.spinner("Processing your sandwich with advanced AI..."):
            score, _, _, _, analysis, is_sandwich, filling_desc = \
                evaluate_symmetry_and_components(image, uploaded_file.type)
        if not is_sandwich:
            st.error("🚫 That doesn't look like a sandwich.")
        else:
            if score >= 90:
                st.balloons() 

            with st.spinner("Generating critique..."):
                comment = generate_comment(score, analysis, filling_desc)

            st.markdown(f"""
            <div class="symmetry-box {'high-score' if score >= 80 else ''}">
                <h3>📊 Symmetry Score<br><span>{score}</span> / 100</h3>
            </div>""", unsafe_allow_html=True)
            
            st.info(f"**AI's Overall Analysis:** {analysis}")
            st.success(f"**Critique:** {comment}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            add_history_entry(st.session_state['user_id'], score, comment, image, timestamp)

    st.markdown("---")
    
    st.subheader("📜 Your Symmetry History")

    user_history = get_user_history(st.session_state['user_id'])

    if not user_history:
        st.write("You haven't analyzed any sandwiches yet. Upload one to start!")
    else:
        for entry in user_history:
            score, comment, thumb_path, timestamp = entry
            st.markdown('<div class="history-entry">', unsafe_allow_html=True)
            cols = st.columns([1, 3])
            with cols[0]:
                if os.path.exists(thumb_path):
                    st.image(thumb_path)
                else:
                    st.error("Image not found")
            with cols[1]:
                st.metric(label="Symmetry Score", value=f"{score}/100")
                st.write(f"**Critique:** {comment}")
                st.caption(f"Analyzed on: {timestamp}")
            st.markdown('</div>', unsafe_allow_html=True)

# --- ROUTING LOGIC ---
if not st.session_state.get('logged_in'):
    if st.session_state.get('page') == 'Signup':
        show_signup_page()
    else:
        show_login_page()
else:
    show_main_app()