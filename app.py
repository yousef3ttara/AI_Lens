"""
AI Lens - Plant Leaf Disease Detection Web Application
A College Machine Learning & Agronomic Advisory Web Platform built with Streamlit.
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st

# Local modules
import disease_info
import utils

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Lens — Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, presentation-grade aesthetic
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1b4d3e 0%, #2e7d32 50%, #43a047 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
    }
    .main-header h1 {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .main-header p {
        font-size: 1.15rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 300;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.8rem;
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Result card styles */
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #2e7d32;
        background-color: #f8faf8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .status-card.diseased {
        border-left-color: #d32f2f;
        background-color: #fff9f9;
    }
    .status-badge-healthy {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 1.05rem;
    }
    .status-badge-diseased {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 1.05rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .info-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .disclaimer-box {
        background-color: #fefce8;
        border: 1px solid #fef08a;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 0.88rem;
        color: #854d0e;
        line-height: 1.5;
        margin-top: 1.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_deep_learning_model():
    """
    Safely load trained MobileNetV2 model if available.
    Returns model object or None. Does not throw runtime crashes.
    """
    model_path = utils.get_available_model_path()
    if not model_path:
        return None

    try:
        import keras
        return keras.models.load_model(model_path)
    except Exception:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.sidebar.error(f"Error loading model weights from {model_path}: {e}")
            return None


@st.cache_data
def get_class_list():
    """Load canonical list of 38 PlantVillage classes."""
    try:
        return utils.load_class_names()
    except Exception:
        # Fallback to keys from disease_info database
        return list(disease_info.DISEASE_DATABASE.keys())


def render_header():
    """Display the title banner, college project header, and project badges."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🔬 AI Lens — Plant Leaf Disease Detection</h1>
            <p>Computer Vision & Transfer Learning System for Botanical Health Screening & Agronomic Advisory</p>
            <div class="badge">College Capstone Project &bull; MobileNetV2 &bull; PlantVillage 38 Classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(is_model_loaded: bool, class_names: list):
    """Render interactive controls, project stats, and scientific disclaimers in sidebar."""
    st.sidebar.image(
        "https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?auto=format&fit=crop&w=600&q=80",
        caption="AI Lens Computer Vision System",
        use_container_width=True,
    )
    st.sidebar.title("System Controls")

    # Hardware & Model Status
    st.sidebar.subheader("Model Status")
    if is_model_loaded:
        st.sidebar.success("✓ MobileNetV2 weights loaded into memory")
        model_source = "Trained Model (`models/plant_disease_mobilenetv2.keras`)"
    else:
        st.sidebar.warning("⚠ Model weights not found in `models/`")
        model_source = "Simulation & Educational Demo Mode"
    st.sidebar.caption(f"Active Engine: **{model_source}**")

    # Confidence Threshold configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("Diagnostics Configuration")
    threshold = st.sidebar.slider(
        "Confidence Alert Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Predictions below this confidence level trigger an uncertainty alert for field verification.",
    )

    # Dataset Specifications
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Metadata")
    st.sidebar.markdown(
        f"""
        - **Dataset**: [PlantVillage (Mohanty et al.)](https://huggingface.co/datasets/mohanty/PlantVillage)
        - **Crops Monitored**: 14 Species
        - **Classes Screened**: {len(class_names)} Categories
        - **Architecture**: MobileNetV2 (ImageNet)
        - **Input Resolution**: 224 &times; 224 RGB
        """
    )

    # Demo Simulation Class selector (when model weights aren't present or user wants to demo specific classes)
    demo_class = None
    if not is_model_loaded:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Demo Class Simulator")
        st.sidebar.info(
            "Select any of the 38 PlantVillage classes to simulate diagnosis, symptoms, and care plans."
        )
        demo_class = st.sidebar.selectbox("Simulate Condition:", class_names, index=29)  # Tomato Early Blight default

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "AI Lens College Project &bull; Built with TensorFlow, Keras & Streamlit"
    )

    return threshold, demo_class


def render_setup_guide():
    """Displayed when model weights are not yet present."""
    with st.expander("ℹ️ Model Setup & Training Instructions (Click to expand)", expanded=True):
        st.markdown(
            """
            ### Welcome to AI Lens!
            Model weights (`models/plant_disease_mobilenetv2.keras`) were not detected in the local repository. 
            The web application is fully initialized and operating in **Educational Simulation Mode** so you can 
            demonstrate the user interface, leaf previews, and care recommendations right away.

            #### How to Train & Load the Real Model:
            1. **Train on Google Colab (Recommended - Free GPU)**:
               - Open the provided notebook: [`notebooks/training.ipynb`](notebooks/training.ipynb).
               - Select **Runtime &rarr; Change runtime type &rarr; T4 GPU**.
               - Run all cells. The dataset will be downloaded directly from Hugging Face (`mohanty/PlantVillage`).
               - Download the resulting `plant_disease_mobilenetv2.keras` and save it to the local `models/` directory.
            2. **Train Locally with CPU / GPU**:
               ```bash
               # Download dataset or run with Hugging Face integration:
               python train.py --use_hf --epochs 15 --batch_size 32
               ```
            3. Once `models/plant_disease_mobilenetv2.keras` is in place, simply refresh this browser tab!
            """
        )


def main():
    render_header()

    class_names = get_class_list()
    model = load_deep_learning_model()
    is_model_loaded = model is not None

    threshold, demo_class = render_sidebar(is_model_loaded, class_names)

    if not is_model_loaded:
        render_setup_guide()

    # Main layout: Two columns (Upload & Preview | Analysis & Care Plan)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Upload Leaf Photograph")
        st.markdown(
            "Upload a clear, focused photograph of a single crop leaf (top or bottom surface)."
        )

        uploaded_file = st.file_uploader(
            "Select an image file (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="High-contrast images with even lighting provide the most reliable leaf texture classification.",
        )

        image_to_analyze = None

        if uploaded_file is not None:
            try:
                image_to_analyze = Image.open(uploaded_file)
                st.image(
                    image_to_analyze,
                    caption=f"Uploaded: {uploaded_file.name} ({image_to_analyze.width} × {image_to_analyze.height})",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Unable to read image file: {e}")
                image_to_analyze = None
        else:
            # Provide sample guidance placeholder
            st.info(
                "📸 **Tip for best results:** Ensure the plant leaf occupies most of the frame, "
                "with clear focus on spots, pustules, or lesions."
            )

        analyze_button = st.button(
            "🔍 Analyze Leaf Condition",
            type="primary",
            use_container_width=True,
            disabled=(image_to_analyze is None and is_model_loaded),
        )

    with col2:
        st.subheader("2. Diagnostic Results & Care Plan")

        if not analyze_button and image_to_analyze is None:
            st.markdown(
                """
                <div style="padding: 2.5rem; text-align: center; border: 2px dashed #cbd5e1; border-radius: 12px; color: #64748b; margin-top: 1rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🍃</div>
                    <div style="font-weight: 600; font-size: 1.1rem;">Awaiting Leaf Image</div>
                    <div style="font-size: 0.9rem; margin-top: 0.25rem;">Upload a leaf photo on the left and click <b>Analyze Leaf Condition</b>.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Perform Analysis
        if analyze_button or image_to_analyze is not None:
            with st.spinner("Processing image and executing MobileNetV2 neural inference..."):
                if is_model_loaded and image_to_analyze is not None:
                    # Real Deep Learning Inference
                    results = utils.predict_leaf(
                        model=model,
                        image=image_to_analyze,
                        class_names=class_names,
                        top_k=3,
                        confidence_threshold=threshold,
                    )
                    top_class = results["top_class"]
                    confidence = results["confidence"]
                    percentage = results["percentage"]
                    top_predictions = results["top_predictions"]
                    is_low_confidence = results["is_low_confidence"]
                    is_ambiguous = results["is_ambiguous"]
                    possible_non_plant = results["possible_non_plant"]
                else:
                    # Simulated Diagnostic for demo / presentation when weights not yet placed
                    simulated_target = demo_class or (class_names[29] if len(class_names) > 29 else class_names[0])
                    top_class = simulated_target
                    confidence = 0.942
                    percentage = 94.2
                    # Create plausible top 3
                    top_predictions = [
                        {"class_name": top_class, "confidence": 0.942, "percentage": 94.2},
                        {"class_name": class_names[30] if len(class_names) > 30 else class_names[1], "confidence": 0.041, "percentage": 4.1},
                        {"class_name": class_names[32] if len(class_names) > 32 else class_names[2], "confidence": 0.012, "percentage": 1.2},
                    ]
                    is_low_confidence = confidence < threshold
                    is_ambiguous = False
                    possible_non_plant = False

            # Retrieve rich botanical & treatment data from knowledge base
            info = disease_info.get_disease_info(top_class)
            is_healthy = info["is_healthy"]

            # Visual Result Card
            status_class = "healthy" if is_healthy else "diseased"
            status_badge_html = (
                """<span class="status-badge-healthy">✓ HEALTHY FOLIAGE</span>"""
                if is_healthy
                else """<span class="status-badge-diseased">⚠️ DISEASE DETECTED</span>"""
            )

            st.markdown(
                f"""
                <div class="status-card {status_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <div>
                            <span class="metric-label">Diagnosis Assessment</span>
                            <h2 style="margin: 0.2rem 0; color: #0f172a;">{info['condition']}</h2>
                        </div>
                        {status_badge_html}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; border-top: 1px solid #e2e8f0; padding-top: 0.8rem;">
                        <div>
                            <div class="metric-label">Plant Species</div>
                            <div style="font-weight: 700; color: #1e293b; font-size: 1.05rem;">{info['species']}</div>
                        </div>
                        <div>
                            <div class="metric-label">Model Confidence Score</div>
                            <div class="metric-value">{percentage:.1f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Uncertainty & Out-of-Distribution Warning
            if is_low_confidence or possible_non_plant or is_ambiguous:
                warning_messages = []
                if is_low_confidence:
                    warning_messages.append(
                        f"• Top confidence score ({percentage:.1f}%) is below the alert threshold ({threshold * 100:.0f}%)."
                    )
                if possible_non_plant:
                    warning_messages.append(
                        "• Image color distribution or contrast does not strongly match typical plant foliage. Verify that the image depicts a real leaf."
                    )
                if is_ambiguous:
                    warning_messages.append(
                        "• The confidence margin between the top two candidate classes is narrow. Symptoms may overlap multiple conditions."
                    )

                st.warning(
                    "⚠️ **Diagnostic Uncertainty Alert:**\n\n"
                    + "\n".join(warning_messages)
                    + "\n\nPlease inspect the specimen closely, retake the photo under natural daylight, or consult a local agronomist."
                )

            # Top 3 Predicted Classes Distribution
            st.markdown("#### Top Model Predictions")
            for pred in top_predictions:
                c_info = disease_info.get_disease_info(pred["class_name"])
                label_display = f"{c_info['species'].split('(')[0].strip()} — {c_info['condition']}"
                st.write(f"**{label_display}**: {pred['percentage']:.1f}%")
                st.progress(min(1.0, max(0.0, pred["confidence"])))

            st.markdown("---")

            # Detailed Agronomic Care & Treatment Recommendations
            st.markdown("### 📋 Agronomic Recommendations & Treatment Plan")

            tab1, tab2, tab3 = st.tabs(["🔍 Symptoms & Pathogen", "🌱 Cultural Management", "💊 Treatment & Prevention"])

            with tab1:
                st.markdown(f"**Pathogen / Causal Agent:** `{info.get('pathogen', 'N/A')}`")
                st.markdown(f"**Severity Level:** `{info.get('severity', 'N/A')}`")
                st.markdown(f"**Observed Symptoms:**\n{info.get('symptoms', 'None reported.')}")

            with tab2:
                st.markdown(f"**Sanitation & Cultural Control:**\n{info.get('cultural_control', 'Standard field sanitation.')}")

            with tab3:
                st.markdown(f"**Recommended Treatments:**\n{info.get('treatment', 'No chemical intervention required.')}")
                st.markdown(f"**Preventative Measures:**\n{info.get('preventative_care', 'Routine monitoring.')}")

    # Prominent Disclaimers (Required by Section 1 & 4)
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>⚠️ Important Agronomic & Scientific Disclaimers:</b><br/>
            1. <b>Not a Professional Diagnosis:</b> Predictions generated by AI Lens are computational pattern-matching estimates based on digital photography and are not guaranteed agricultural or botanical diagnoses. Always consult an accredited plant pathologist or agricultural extension officer before taking critical crop remediation decisions.<br/>
            2. <b>Softmax Confidence:</b> Neural network softmax scores reflect relative class probabilities across the 38 trained PlantVillage classes; they are not fully calibrated statistical probabilities.<br/>
            3. <b>Unfamiliar & Non-Leaf Images:</b> The model is trained exclusively on 14 crop species within the PlantVillage dataset. Images of unsupported plant varieties, non-leaf objects, or poor-lighting photographs may result in unreliable predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
