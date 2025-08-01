import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
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
    generation_config = genai.types.GenerationConfig(temperature=0)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash', generation_config=generation_config)
    vision_model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)
except Exception as e:
    st.error(f"Configuration Error: Could not load Gemini API key. Details: {e}")
    st.stop()

# --- Session State Management ---
if 'logged_in' not in st.session_state:
    st.session_state.update({
        'logged_in': False, 'user_id': None, 'username': None, 'page': 'Login'
    })
if 'glow_active' not in st.session_state:
    st.session_state.glow_active = False

# --- UI Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        html, body, *, h1, h2, h3, h4, h5, h6, [class*="st-"] {
            font-family: 'Poppins', sans-serif !important;
        }
        .stApp {
            background: linear-gradient(to right, #16222A, #3A6073);
        }
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
            transition: filter 0.5s ease-in-out;
            filter: drop-shadow(0 5px 15px rgba(0,0,0,0.3));
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
            transition: all 0.5s ease;
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
        .symmetry-box, .history-entry {
            background-color: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 25px;
        }
        .history-entry { margin-bottom: 15px; }
        .glow-effect .main-title {
            background: none;
            -webkit-text-fill-color: white;
            text-shadow: 0 0 8px #fff, 0 0 16px #fff, 0 0 24px #fff;
        }
        .glow-effect .logo-emoji {
            filter: drop-shadow(0 0 15px rgba(255, 255, 255, 0.8));
        }
    </style>
""", unsafe_allow_html=True)


# --- Helper and Core Logic Functions ---
def get_image_bytes_and_mime(image_pil, uploaded_file_type):
    image_bytes_io = io.BytesIO()
    image_pil.save(image_bytes_io, format=image_pil.format or 'PNG')
    return image_bytes_io.getvalue(), uploaded_file_type

def get_sandwich_bounding_box(image_bytes, mime_type, img_width, img_height):
    try:
        prompt = "Identify the main sandwich... Provide its bounding box..."
        response = vision_model.generate_content([prompt, {'mime_type': mime_type, 'data': image_bytes}])
        match = re.search(r'(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)', response.text.strip())
        if match:
            x_min_perc, y_min_perc, x_max_perc, y_max_perc = map(float, match.groups())
            x_min, y_min, x_max, y_max = int(x_min_perc/100*img_width), int(y_min_perc/100*img_height), int(x_max_perc/100*img_width), int(y_max_perc/100*img_height)
            if x_min > x_max: x_min, x_max = x_max, x_min
            if y_min > y_max: y_min, y_max = y_max, y_min
            padding_x, padding_y = int(img_width * 0.02), int(img_height * 0.02)
            x_min, y_min, x_max, y_max = max(0, x_min - padding_x), max(0, y_min - padding_y), min(img_width, x_max + padding_x), min(img_height, y_max + padding_y)
            return (x_min, y_min, x_max, y_max)
        return None
    except Exception:
        return None

def analyze_filling_symmetry(cropped_image_bytes, mime_type):
    try:
        prompt = "This is an image of a sandwich... Describe how evenly... the filling ingredients are distributed..."
        response = vision_model.generate_content([prompt, {'mime_type': mime_type, 'data': cropped_image_bytes}])
        return response.text.strip()
    except Exception:
        return "AI could not analyze filling symmetry."

def evaluate_symmetry_and_components(image_pil, uploaded_file_type):
    st.info("Detecting sandwich boundaries with AI...")
    original_width, original_height = image_pil.size
    image_bytes, mime_type = get_image_bytes_and_mime(image_pil, uploaded_file_type)
    bbox = get_sandwich_bounding_box(image_bytes, mime_type, original_width, original_height)
    if bbox and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
        cropped_image_pil = image_pil.crop(bbox)
    else:
        st.warning("Could not automatically crop. Analyzing full image.")
        cropped_image_pil = image_pil
    image_pil_resized = cropped_image_pil.resize((400, 400))
    image_gray = image_pil_resized.convert("L")
    image_blurred = image_gray.filter(ImageFilter.GaussianBlur(radius=2))
    img_np = np.array(image_blurred)
    mid = img_np.shape[1] // 2
    left, right = img_np[:, :mid], img_np[:, mid:]
    right_flipped = np.fliplr(right)
    min_w = min(left.shape[1], right_flipped.shape[1])
    left, right_flipped = left[:, :min_w], right_flipped[:, :min_w]
    diff = np.abs(left.astype(np.int16) - right_flipped.astype(np.int16))
    score = round(100 - (np.mean(diff) / 255 * 100), 2)
    is_actually_sandwich = "yes" in vision_model.generate_content(["Is this a sandwich? Yes or No.", {'mime_type': mime_type, 'data': image_bytes}]).text.strip().lower()
    if is_actually_sandwich:
        ai_sandwich_analysis = vision_model.generate_content(["Describe the main food item...", {'mime_type': mime_type, 'data': image_bytes}]).text.strip()
        filling_analysis_description = analyze_filling_symmetry(*get_image_bytes_and_mime(cropped_image_pil, uploaded_file_type))
    else:
        ai_sandwich_analysis = "AI did not identify a sandwich."
        filling_analysis_description = "N/A"
    return score, None, None, None, ai_sandwich_analysis, is_actually_sandwich, filling_analysis_description

def generate_comment(score, ai_sandwich_analysis, filling_analysis_description):
    prompt = f"You are a sarcastic food critic... A sandwich scored {score}/100... Write a one-line review..."
    try:
        return gemini_model.generate_content(prompt).text.strip()
    except Exception:
        return f"A respectable {score}/100."

# --- UI Functions ---
def show_login_page():
    st.header("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.update({'logged_in': True, 'user_id': user_id, 'username': username})
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
                st.error("Username already taken.")
    if st.button("Already have an account? Login"):
        st.session_state['page'] = 'Login'
        st.rerun()

# --- MAIN APP DISPLAY ---
def show_main_app():
    st.sidebar.header(f"Welcome, {st.session_state['username']}!")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Title Effect")
    dark_col, light_col = st.sidebar.columns(2)
    if dark_col.button("🌙 Dark mode", use_container_width=True):
        st.session_state.glow_active = False
        st.rerun()
    if light_col.button("☀️ Light mode", use_container_width=True):
        st.session_state.glow_active = True
        st.rerun()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    glow_class = "glow-effect" if st.session_state.get('glow_active', False) else ""
    st.markdown(f"""
        <div class="title-container {glow_class}">
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
            score, _, _, _, analysis, is_sandwich, filling_desc = evaluate_symmetry_and_components(image, uploaded_file.type)
        if not is_sandwich:
            st.error("🚫 That doesn't look like a sandwich.")
        else:
            if score >= 90:
                st.balloons()
            with st.spinner("Generating critique..."):
                comment = generate_comment(score, analysis, filling_desc)
            st.markdown(f"""
            <div class="symmetry-box">
                <h3>📊 Symmetry Score<br><span>{score}</span> / 100</h3>
            </div>""", unsafe_allow_html=True)
            st.info(f"**AI's Overall Analysis:** {analysis}")
            st.success(f"**Critique:** {comment}")
            add_history_entry(st.session_state['user_id'], score, comment, image, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    st.markdown("---")
    st.subheader("📜 Your Symmetry History")
    user_history = get_user_history(st.session_state['user_id'])
    if not user_history:
        st.write("You haven't analyzed any sandwiches yet.")
    else:
        for entry in user_history:
            score, comment, thumb_path, timestamp = entry
            st.markdown('<div class="history-entry">', unsafe_allow_html=True)
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(thumb_path if os.path.exists(thumb_path) else "Image not found")
            with cols[1]:
                st.metric(label="Symmetry Score", value=f"{score}/100")
                st.write(f"**Critique:** {comment}")
                st.caption(f"Analyzed on: {timestamp}")
            st.markdown('</div>', unsafe_allow_html=True)

# --- ROUTING LOGIC ---
if not st.session_state.get('logged_in'):
    if st.session_state.get('page', 'Login') == 'Login':
        show_login_page()
    else:
        show_signup_page()
else:
    show_main_app()