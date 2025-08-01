# database.py

import sqlite3
import hashlib
import os
from PIL import Image
import io

# --- Constants ---
DB_NAME = "sandwich_app.db"
IMAGE_DIR = "user_images"

# --- Database Setup ---
def setup_database():
    """Initializes the database and creates tables if they don't exist."""
    # Create image directory if it doesn't exist
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    
    # Create history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            score REAL NOT NULL,
            comment TEXT,
            image_path TEXT,
            thumbnail_path TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()

# --- Password Utilities ---
def hash_password(password):
    """Hashes a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    """Verifies a provided password against a stored hash."""
    return stored_hash == hashlib.sha256(provided_password.encode()).hexdigest()

# --- User Management ---
def create_user(username, password):
    """Creates a new user. Returns True on success, False on failure (e.g., user exists)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError: # This error occurs if username is not unique
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticates a user. Returns user_id on success, None on failure."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    user_record = cursor.fetchone()
    conn.close()
    
    if user_record and verify_password(user_record[1], password):
        return user_record[0]  # Return user_id
    return None

# --- History Management ---
def save_image_and_get_paths(image_pil, user_id, timestamp_str):
    """Saves full image and thumbnail, returning their paths."""
    # Sanitize timestamp for filename
    safe_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
    
    # Save original image
    image_filename = f"user_{user_id}_{safe_timestamp}.png"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    image_pil.save(image_path, "PNG")

    # Create and save thumbnail
    thumbnail = image_pil.copy()
    thumbnail.thumbnail((120, 120))
    thumb_filename = f"thumb_{image_filename}"
    thumb_path = os.path.join(IMAGE_DIR, thumb_filename)
    thumbnail.save(thumb_path, "PNG")

    return image_path, thumb_path


def add_history_entry(user_id, score, comment, image_pil, timestamp):
    """Adds a new analysis entry to the database for a specific user."""
    
    image_path, thumb_path = save_image_and_get_paths(image_pil, user_id, timestamp)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO history (user_id, score, comment, image_path, thumbnail_path, timestamp) 
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, score, comment, image_path, thumb_path, timestamp)
    )
    conn.commit()
    conn.close()

def get_user_history(user_id):
    """Retrieves all history entries for a specific user, newest first."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT score, comment, thumbnail_path, timestamp FROM history WHERE user_id = ? ORDER BY id DESC",
        (user_id,)
    )
    history = cursor.fetchall()
    conn.close()
    return history