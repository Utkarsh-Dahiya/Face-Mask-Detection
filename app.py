"""
Face Mask Detection — Streamlit Application
=============================================
A production-ready Streamlit UI wrapping a MobileNetV2-based
binary classifier (face_mask_detector.keras).

Model notes (do not change without retraining):
- Input size expected by the model: 160x160x3
- The model itself contains a `Rescaling(1./255)` layer, so raw
  0-255 pixel arrays must be passed in (no manual normalization).
- Single sigmoid output. Training class order was:
    ['WithMask', 'WithoutMask']
  i.e. prediction > 0.5  -> "WithoutMask"
       prediction <= 0.5 -> "WithMask"
"""

import io
import time
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# TensorFlow is imported lazily inside load_model() so the rest of the
# app (page config, CSS, etc.) renders even if TF import is slow.


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODEL_PATH = "face_mask_detector.keras"
IMG_SIZE = (160, 160)                     # must match training pipeline
CLASS_NAMES = ["WithMask", "WithoutMask"]  # alphabetical order from training
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_FILE_MB = 10

APP_TITLE = "AI Face Mask Detector"
AUTHOR_NAME = "Your Name Here"
GITHUB_URL = "https://github.com/your-username"
LINKEDIN_URL = "https://linkedin.com/in/your-profile"


# --------------------------------------------------------------------------- #
# Page configuration (must be first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Custom CSS — premium dark theme
# --------------------------------------------------------------------------- #
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 20% 20%, #131722 0%, #0b0e14 55%, #05060a 100%);
        }

        /* Hide default streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ---------- Hero ---------- */
        .hero-container {
            padding: 2.2rem 2rem 1.8rem 2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(16,185,129,0.10) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.6rem;
            text-align: center;
            animation: fadeInDown 0.8s ease-out;
        }

        .gradient-title {
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #6366f1, #22d3ee, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
        }

        .hero-subtitle {
            color: #9ca3af;
            font-size: 1.05rem;
            font-weight: 400;
            margin-top: 0.6rem;
        }

        .badge-row {
            display: flex;
            justify-content: center;
            gap: 0.6rem;
            margin-top: 1.1rem;
            flex-wrap: wrap;
        }

        .badge {
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            color: #d1d5db;
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-18px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }

        /* ---------- Cards ---------- */
        .glass-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
            animation: fadeIn 0.6s ease-out;
        }

        .result-card-mask {
            background: linear-gradient(135deg, rgba(16,185,129,0.16), rgba(16,185,129,0.03));
            border: 1px solid rgba(16,185,129,0.4);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            animation: fadeIn 0.5s ease-out;
        }

        .result-card-nomask {
            background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.04));
            border: 1px solid rgba(239,68,68,0.45);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            animation: fadeIn 0.5s ease-out;
        }

        .result-label {
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0.3rem 0;
        }

        .result-sub {
            color: #d1d5db;
            font-size: 0.95rem;
        }

        .conf-number {
            font-size: 2.6rem;
            font-weight: 800;
            margin: 0.2rem 0;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1017 0%, #090b10 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .sidebar-info-row {
            display: flex;
            justify-content: space-between;
            padding: 0.45rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 0.88rem;
        }
        .sidebar-info-row span:first-child { color: #9ca3af; }
        .sidebar-info-row span:last-child { color: #e5e7eb; font-weight: 600; }

        /* ---------- Buttons ---------- */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            padding: 0.7rem 1rem;
            font-weight: 600;
            font-size: 1rem;
            background: linear-gradient(90deg, #6366f1, #22d3ee);
            color: white;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(99,102,241,0.35);
            color: white;
        }

        /* Secondary / reset button variant */
        div[data-testid="stHorizontalBlock"] .stButton:nth-of-type(2) > button {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.15);
        }

        /* ---------- Footer ---------- */
        .app-footer {
            margin-top: 3rem;
            padding: 1.6rem 1rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            text-align: center;
            color: #6b7280;
            font-size: 0.85rem;
        }
        .app-footer a {
            color: #9ca3af;
            text-decoration: none;
            margin: 0 0.6rem;
            font-weight: 500;
        }
        .app-footer a:hover { color: #22d3ee; }

        /* History pills */
        .history-pill {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 10px;
            font-size: 0.8rem;
            margin: 0.2rem;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load the trained Keras model once and cache it across reruns."""
    import tensorflow as tf  # local import keeps startup snappy
    model = tf.keras.models.load_model(model_path)
    return model


# --------------------------------------------------------------------------- #
# Image validation & preprocessing
# --------------------------------------------------------------------------- #
def validate_upload(uploaded_file) -> str | None:
    """Return an error message if the upload is invalid, else None."""
    if uploaded_file is None:
        return "No file uploaded."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return f"File is too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_MB} MB."

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext not in ALLOWED_TYPES:
        return f"Unsupported file type '.{ext}'. Please upload JPG, JPEG, or PNG."

    return None


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize + format an image exactly as the model expects.

    The model has an internal Rescaling(1./255) layer, so we feed
    raw 0-255 float pixel values — matching image_dataset_from_directory
    behaviour used at training time.
    """
    image = pil_image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image).astype("float32")   # 0-255 range, no /255 here
    array = np.expand_dims(array, axis=0)          # (1, 160, 160, 3)
    return array


def predict(model, pil_image: Image.Image) -> dict:
    """Run inference and return a structured result dict."""
    batch = preprocess_image(pil_image)

    start = time.time()
    raw_pred = model.predict(batch, verbose=0)
    elapsed = time.time() - start

    score = float(raw_pred[0][0])  # sigmoid output in [0, 1]
    label_idx = int(score > 0.5)
    label = CLASS_NAMES[label_idx]

    confidence = score if label_idx == 1 else (1.0 - score)

    return {
        "label": label,
        "confidence": confidence,
        "raw_score": score,
        "inference_time": elapsed,
    }


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <div class="gradient-title">😷 AI Face Mask Detector</div>
            <div class="hero-subtitle">
                Real-time face mask classification powered by MobileNetV2 &amp; Transfer Learning
            </div>
            <div class="badge-row">
                <span class="badge">⚡ TensorFlow / Keras</span>
                <span class="badge">🧠 MobileNetV2 Backbone</span>
                <span class="badge">🎯 Binary Classification</span>
                <span class="badge">🖼️ 160×160 Input</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📋 Project Information")
        st.markdown(
            f"""
            <div class="sidebar-info-row"><span>Model</span><span>MobileNetV2</span></div>
            <div class="sidebar-info-row"><span>Framework</span><span>TensorFlow</span></div>
            <div class="sidebar-info-row"><span>Problem Type</span><span>Binary Classification</span></div>
            <div class="sidebar-info-row"><span>Image Size</span><span>{IMG_SIZE[0]}×{IMG_SIZE[1]}</span></div>
            <div class="sidebar-info-row"><span>Classes</span><span>WithMask / WithoutMask</span></div>
            <div class="sidebar-info-row"><span>Author</span><span>{AUTHOR_NAME}</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### ⚙️ Session Controls")
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.history = []
            st.toast("History cleared.", icon="✅")

        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        history = st.session_state.get("history", [])
        st.metric("Predictions this session", len(history))
        if history:
            mask_count = sum(1 for h in history if h["label"] == "WithMask")
            st.metric("With Mask detected", mask_count)
            st.metric("Without Mask detected", len(history) - mask_count)


def render_upload_area():
    st.markdown("#### 📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Drag and drop a face image here, or click to browse",
        type=ALLOWED_TYPES,
        help=f"Supported formats: {', '.join(t.upper() for t in ALLOWED_TYPES)} · Max {MAX_FILE_MB} MB",
        label_visibility="collapsed",
    )
    return uploaded_file


def render_result_card(result: dict) -> None:
    label = result["label"]
    confidence_pct = result["confidence"] * 100

    if label == "WithMask":
        st.markdown(
            f"""
            <div class="result-card-mask">
                <div style="font-size:3rem;">✅</div>
                <div class="result-label" style="color:#34d399;">Mask Detected</div>
                <div class="result-sub">The person appears to be wearing a face mask.</div>
                <div class="conf-number" style="color:#34d399;">{confidence_pct:.1f}%</div>
                <div class="result-sub">Confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-nomask">
                <div style="font-size:3rem;">⚠️</div>
                <div class="result-label" style="color:#f87171;">No Mask Detected</div>
                <div class="result-sub">The person does not appear to be wearing a face mask.</div>
                <div class="conf-number" style="color:#f87171;">{confidence_pct:.1f}%</div>
                <div class="result-sub">Confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Confidence Meter**")
    st.progress(min(max(result["confidence"], 0.0), 1.0))

    st.caption(
        f"⏱️ Inference time: {result['inference_time']*1000:.1f} ms · "
        f"Raw sigmoid score: {result['raw_score']:.4f}"
    )

    explanation = (
        "The model detected facial features consistent with a mask covering the "
        "nose and mouth region, resulting in a high confidence classification."
        if label == "WithMask"
        else
        "The model did not detect the visual patterns typically associated with "
        "a worn face mask, such as occluded nose/mouth regions."
    )
    st.info(f"ℹ️ {explanation}")


def render_history() -> None:
    history = st.session_state.get("history", [])
    if not history:
        return

    st.markdown("#### 🕘 Prediction History (this session)")
    for i, item in enumerate(reversed(history[-10:])):
        icon = "✅" if item["label"] == "WithMask" else "⚠️"
        st.markdown(
            f"""<span class="history-pill">{icon} {item['label']} — """
            f"""{item['confidence']*100:.1f}% · {item['timestamp']}</span>""",
            unsafe_allow_html=True,
        )


def render_learn_more() -> None:
    st.markdown("### 📚 Learn More")

    with st.expander("🔍 How does this model work?"):
        st.markdown(
            """
            The uploaded image is resized to **160×160 pixels** and passed
            through a convolutional neural network. The network outputs a
            single probability between 0 and 1 (a *sigmoid* score) indicating
            how likely the image is to belong to the **"Without Mask"** class.
            A threshold of **0.5** is used to decide the final label, and the
            distance from that threshold becomes the reported confidence.
            """
        )

    with st.expander("🔁 What is Transfer Learning?"):
        st.markdown(
            """
            Transfer learning reuses a model that was already trained on a
            large, general dataset (in this case **ImageNet**, containing
            millions of images across thousands of categories) and adapts it
            to a new, more specific task — here, distinguishing masked from
            unmasked faces. This saves training time and works well even with
            a relatively small, specialized dataset, since the network
            already understands general visual patterns like edges,
            textures, and shapes.
            """
        )

    with st.expander("📱 What is MobileNetV2?"):
        st.markdown(
            """
            **MobileNetV2** is a lightweight convolutional neural network
            architecture designed for mobile and embedded devices. It uses
            *depthwise separable convolutions* and *inverted residual blocks*
            to drastically reduce the number of parameters and computations
            compared to traditional CNNs, while retaining strong accuracy.
            This makes it an efficient backbone for real-time applications
            such as this face mask detector.
            """
        )


def render_footer() -> None:
    year = datetime.now().year
    st.markdown(
        f"""
        <div class="app-footer">
            <a href="{GITHUB_URL}" target="_blank">🐙 GitHub</a> ·
            <a href="{LINKEDIN_URL}" target="_blank">💼 LinkedIn</a>
            <br><br>
            © {year} {AUTHOR_NAME} · Built with Streamlit &amp; TensorFlow
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_custom_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    render_hero()
    render_sidebar()

    # Load model (cached) with a friendly spinner on first run only
    try:
        with st.spinner("🔄 Loading AI model..."):
            model = load_model(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "❌ Failed to load the model. Make sure "
            f"`{MODEL_PATH}` is present in the app directory.\n\n"
            f"Details: {exc}"
        )
        st.stop()

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        uploaded_file = render_upload_area()

        if uploaded_file is not None:
            error_msg = validate_upload(uploaded_file)
            if error_msg:
                st.error(f"❌ {error_msg}")
            else:
                try:
                    image_bytes = uploaded_file.getvalue()
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    pil_image.verify()  # sanity check it's a real image
                    pil_image = Image.open(io.BytesIO(image_bytes))  # reopen after verify
                    st.session_state.current_image = pil_image
                except UnidentifiedImageError:
                    st.error("❌ This file could not be read as an image. Please try another file.")
                    st.session_state.current_image = None
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ Unexpected error reading image: {exc}")
                    st.session_state.current_image = None

        if st.session_state.current_image is not None:
            st.markdown("#### 🖼️ Preview")
            st.image(st.session_state.current_image, use_container_width=True)

            btn_col1, btn_col2 = st.columns(2)
            predict_clicked = btn_col1.button("🚀 Predict")
            reset_clicked = btn_col2.button("♻️ Reset")

            if reset_clicked:
                st.session_state.current_image = None
                st.rerun()

            if predict_clicked:
                with st.spinner("🧠 Analyzing image..."):
                    try:
                        result = predict(model, st.session_state.current_image)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"❌ Prediction failed: {exc}")
                        result = None

                if result:
                    result["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    st.session_state.history.append(result)
                    st.session_state.last_result = result

    with right_col:
        st.markdown("#### 🎯 Prediction Result")
        if st.session_state.get("last_result") and st.session_state.current_image is not None:
            render_result_card(st.session_state.last_result)
        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; color:#9ca3af;">
                    <div style="font-size:2.4rem;">🤖</div>
                    Upload an image and click <b>Predict</b> to see results here.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    render_history()

    st.markdown("---")
    render_learn_more()

    render_footer()


if __name__ == "__main__":
    main()