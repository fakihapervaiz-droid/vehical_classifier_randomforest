import streamlit as st
import pickle
import cv2
import numpy as np
import os
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VehicleVision AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: #f5f7fb;
}

/* Main container */
.block-container {
    max-width: 1180px;
    padding-top: 35px;
    padding-bottom: 60px;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
    padding: 50px 55px;
    border-radius: 24px;
    margin-bottom: 30px;
    color: white;
}

.hero-small {
    color: #9ca3af;
    font-size: 14px;
    letter-spacing: 2px;
    font-weight: 700;
    text-transform: uppercase;
}

.hero-title {
    font-size: 52px;
    font-weight: 850;
    line-height: 1.05;
    margin-top: 10px;
    margin-bottom: 15px;
}

.hero-text {
    color: #d1d5db;
    font-size: 18px;
    max-width: 750px;
    line-height: 1.6;
}

/* Section */
.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    margin-top: 35px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #6b7280;
    font-size: 15px;
    margin-bottom: 20px;
}

/* Metric cards */
.metric {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 23px;
    min-height: 125px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.04);
}

.metric-label {
    color: #6b7280;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.metric-value {
    color: #111827;
    font-size: 32px;
    font-weight: 850;
    margin-top: 8px;
}

.metric-note {
    color: #9ca3af;
    font-size: 12px;
    margin-top: 5px;
}

/* White cards */
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.04);
}

/* Upload */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    border: 1px dashed #cbd5e1;
    padding: 10px;
}

/* Prediction */
.prediction-card {
    background: #111827;
    color: white;
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    min-height: 260px;
}

.prediction-label {
    color: #9ca3af;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-name {
    font-size: 38px;
    font-weight: 850;
    margin: 12px 0;
}

.confidence {
    font-size: 17px;
    color: #d1d5db;
}

/* Comparison */
.compare {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 28px;
}

/* Pipeline */
.step {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    min-height: 105px;
}

.step-number {
    font-size: 13px;
    color: #6b7280;
    font-weight: 700;
}

.step-title {
    font-size: 16px;
    font-weight: 750;
    color: #111827;
    margin-top: 8px;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 55px;
    padding-top: 25px;
    border-top: 1px solid #e5e7eb;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "vehicle_random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file not found. Please place "
        "`vehicle_random_forest_model.pkl` in the same folder as app.py."
    )

    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ============================================================
# CLASSES
# ============================================================

classes = [
    "Auto Rickshaws",
    "Bikes",
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains"
]


# ============================================================
# MODEL ACCURACY
# ============================================================

# Your actual Decision Tree result
decision_tree_accuracy = 43.38

# ------------------------------------------------------------
# IMPORTANT:
# Enter the Random Forest accuracy printed by your notebook.
#
# Example:
# random_forest_accuracy = 58.75
# ------------------------------------------------------------

random_forest_accuracy = 0.0


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-small">
Machine Learning • Computer Vision • Classification
</div>

<div class="hero-title">
VehicleVision AI
</div>

<div class="hero-text">
An end-to-end vehicle image classification system that
transforms raw vehicle images into intelligent predictions
using Machine Learning and Random Forest.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# KEY METRICS
# ============================================================

st.markdown(
    '<div class="section-title">Project at a Glance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'From image preprocessing to model deployment'
    '</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">Vehicle Classes</div>
        <div class="metric-value">7</div>
        <div class="metric-note">Multi-class classification</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">Dataset</div>
        <div class="metric-value">5,590</div>
        <div class="metric-note">Vehicle images</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">Baseline</div>
        <div class="metric-value">43.38%</div>
        <div class="metric-note">Decision Tree accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with m4:

    if random_forest_accuracy > 0:
        rf_display = f"{random_forest_accuracy:.2f}%"
    else:
        rf_display = "Pending"

    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">Final Model</div>
        <div class="metric-value">{rf_display}</div>
        <div class="metric-note">Random Forest accuracy</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL EVOLUTION
# ============================================================

st.markdown(
    '<div class="section-title">Model Evolution</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Why the project moved from a single Decision Tree to Random Forest'
    '</div>',
    unsafe_allow_html=True
)

if random_forest_accuracy > 0:

    improvement = (
        random_forest_accuracy -
        decision_tree_accuracy
    )

    relative_improvement = (
        improvement /
        decision_tree_accuracy
    ) * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">Decision Tree</div>
            <div class="metric-value">{decision_tree_accuracy:.2f}%</div>
            <div class="metric-note">Baseline model</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">Random Forest</div>
            <div class="metric-value">{random_forest_accuracy:.2f}%</div>
            <div class="metric-note">Ensemble model</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">Improvement</div>
            <div class="metric-value">+{improvement:.2f}</div>
            <div class="metric-note">
                percentage points ({relative_improvement:.1f}% relative)
            </div>
        </div>
        """, unsafe_allow_html=True)

    comparison = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy": [
            decision_tree_accuracy,
            random_forest_accuracy
        ]
    })

    st.bar_chart(
        comparison.set_index("Model"),
        y="Accuracy"
    )

else:

    st.markdown("""
    <div class="card">

    <b>Decision Tree baseline: 43.38%</b>

    <br><br>

    The first model established a baseline accuracy of
    <b>43.38%</b>. Random Forest was introduced as the
    next step to improve stability and generalization by
    combining multiple decision trees.

    <br><br>

    <b>Run the Random Forest training notebook and enter
    its final accuracy in the app to complete this comparison.</b>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

st.markdown(
    '<div class="section-title">Try the AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a vehicle image and see what the trained model predicts.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    original_img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Preprocess exactly like training
    processed_img = cv2.resize(
        original_img,
        (64, 64)
    )

    flattened = processed_img.flatten().reshape(
        1, -1
    )

    # Prediction
    prediction = model.predict(flattened)

    predicted_index = int(prediction[0])

    predicted_class = classes[predicted_index]

    # Probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            flattened
        )[0]

        confidence = float(
            np.max(probabilities)
        ) * 100

    else:
        probabilities = None
        confidence = 0


    image_col, result_col = st.columns(
        [1.1, 0.9]
    )

    with image_col:

        st.image(
            cv2.cvtColor(
                original_img,
                cv2.COLOR_BGR2RGB
            ),
            caption="Uploaded Image",
            use_container_width=True
        )

    with result_col:

        st.markdown(f"""
        <div class="prediction-card">

            <div class="prediction-label">
                AI Prediction
            </div>

            <div class="prediction-name">
                {predicted_class}
            </div>

            <div class="confidence">
                Confidence: {confidence:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # PROBABILITY BREAKDOWN
    # ========================================================

    if probabilities is not None:

        st.markdown(
            '<div class="section-title">Prediction Breakdown</div>',
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame({
            "Vehicle": classes,
            "Probability": probabilities * 100
        })

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Vehicle")
        )


# ============================================================
# HOW RANDOM FOREST IMPROVES THE MODEL
# ============================================================

st.markdown(
    '<div class="section-title">Why Random Forest?</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:

    st.markdown("""
    <div class="card">

    ### Decision Tree

    The initial model used a single Decision Tree.

    **Accuracy: 43.38%**

    A single tree can become highly dependent on the
    particular training data and may produce unstable
    predictions.

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="card">

    ### Random Forest

    Random Forest combines predictions from many
    Decision Trees.

    Instead of relying on one tree, the ensemble
    considers multiple trees before producing the
    final classification.

    **Result:** More robust and stable predictions.

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">Project Pipeline</div>',
    unsafe_allow_html=True
)

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown("""
    <div class="step">
        <div class="step-number">01</div>
        <div class="step-title">Vehicle Images</div>
    </div>
    """, unsafe_allow_html=True)

with p2:
    st.markdown("""
    <div class="step">
        <div class="step-number">02</div>
        <div class="step-title">Resize 64×64</div>
    </div>
    """, unsafe_allow_html=True)

with p3:
    st.markdown("""
    <div class="step">
        <div class="step-number">03</div>
        <div class="step-title">Pixel Features</div>
    </div>
    """, unsafe_allow_html=True)

with p4:
    st.markdown("""
    <div class="step">
        <div class="step-number">04</div>
        <div class="step-title">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")

p5, p6, p7, p8 = st.columns(4)

with p5:
    st.markdown("""
    <div class="step">
        <div class="step-number">05</div>
        <div class="step-title">Classification</div>
    </div>
    """, unsafe_allow_html=True)

with p6:
    st.markdown("""
    <div class="step">
        <div class="step-number">06</div>
        <div class="step-title">Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)

with p7:
    st.markdown("""
    <div class="step">
        <div class="step-number">07</div>
        <div class="step-title">Pickle Model</div>
    </div>
    """, unsafe_allow_html=True)

with p8:
    st.markdown("""
    <div class="step">
        <div class="step-number">08</div>
        <div class="step-title">Streamlit App</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.markdown(
    '<div class="section-title">Technology Stack</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">

**Python** &nbsp; • &nbsp;
**OpenCV** &nbsp; • &nbsp;
**NumPy** &nbsp; • &nbsp;
**Scikit-learn** &nbsp; • &nbsp;
**Random Forest** &nbsp; • &nbsp;
**Pickle** &nbsp; • &nbsp;
**Streamlit**

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

VehicleVision AI — Machine Learning Portfolio Project

</div>
""", unsafe_allow_html=True)