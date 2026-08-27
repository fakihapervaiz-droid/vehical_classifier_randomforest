import streamlit as st
import pickle
import cv2
import numpy as np
import os
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vehicle Classification AI",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# CUSTOM CSS - PORTFOLIO STYLE
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Main title */
.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 5px;
    color: #111827;
}

.hero-subtitle {
    font-size: 20px;
    color: #6b7280;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.metric-number {
    font-size: 32px;
    font-weight: 800;
    color: #111827;
}

.metric-label {
    color: #6b7280;
    font-size: 15px;
}

/* Prediction */
.prediction {
    background: #111827;
    color: white;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.prediction-title {
    font-size: 17px;
    opacity: 0.8;
}

.prediction-result {
    font-size: 38px;
    font-weight: 800;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    padding: 30px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

MODEL_PATH = "vehicle_random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        "Random Forest model not found. "
        "Please place 'vehicle_random_forest_model.pkl' "
        "in the same folder as app.py."
    )

    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ============================================================
# VEHICLE CLASSES
# ============================================================

categories = [
    "Auto Rickshaws",
    "Bikes",
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains"
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">Vehicle Classification AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'An image classification project using Machine Learning '
    'and Random Forest'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">7</div>
        <div class="metric-label">Vehicle Classes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">64×64</div>
        <div class="metric-label">Image Size</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">100</div>
        <div class="metric-label">Random Forest Trees</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">5,590</div>
        <div class="metric-label">Images</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ============================================================
# IMAGE PREDICTION SECTION
# ============================================================

st.markdown("""
<div class="card">
<div class="card-title">Vehicle Image Classification</div>
<p>
Upload an image and let the trained Random Forest model
identify the vehicle category.
</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Resize
    resized_img = cv2.resize(
        img,
        (64, 64)
    )

    # Flatten
    flattened_img = resized_img.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(flattened_img)

    predicted_class = categories[int(prediction[0])]

    # Confidence
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            flattened_img
        )[0]

        confidence = float(
            np.max(probabilities)
        ) * 100

    else:
        confidence = 0


    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            ),
            caption="Uploaded Vehicle Image",
            use_container_width=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="prediction">

                <div class="prediction-title">
                    Predicted Vehicle
                </div>

                <div class="prediction-result">
                    {predicted_class}
                </div>

                <div style="margin-top:15px;">
                    Confidence: {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.progress(
            min(confidence / 100, 1.0)
        )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    if hasattr(model, "predict_proba"):

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({
            "Vehicle": categories,
            "Probability (%)":
                probabilities * 100
        })

        probability_df = probability_df.sort_values(
            "Probability (%)",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Vehicle")
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.header("Model Performance: Decision Tree → Random Forest")

st.write(
    "The Decision Tree was used as the initial baseline model. "
    "Random Forest was then applied to improve the classification "
    "by combining multiple decision trees."
)


# IMPORTANT:
# Replace this value with the actual Random Forest accuracy
# printed during your training.
decision_tree_accuracy = 43.38

# Change this value after seeing your Random Forest result
random_forest_accuracy = 0.0


if random_forest_accuracy > 0:

    improvement = (
        random_forest_accuracy
        - decision_tree_accuracy
    )

    relative_improvement = (
        improvement / decision_tree_accuracy
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Decision Tree",
            f"{decision_tree_accuracy:.2f}%"
        )

    with col2:

        st.metric(
            "Random Forest",
            f"{random_forest_accuracy:.2f}%"
        )

    with col3:

        st.metric(
            "Accuracy Improvement",
            f"+{improvement:.2f}%"
        )

    st.write("")

    comparison_df = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            decision_tree_accuracy,
            random_forest_accuracy
        ]
    })

    st.bar_chart(
        comparison_df.set_index("Model")
    )

    st.info(
        f"Random Forest improved the accuracy by "
        f"{improvement:.2f} percentage points, "
        f"which represents a relative improvement of "
        f"{relative_improvement:.2f}% over the Decision Tree."
    )

else:

    st.warning(
        "Enter your actual Random Forest accuracy in "
        "`random_forest_accuracy` to display the comparison."
    )


# ============================================================
# WHY RANDOM FOREST?
# ============================================================

st.divider()

st.header("Why Random Forest Performed Better")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    ### Decision Tree

    A Decision Tree uses a **single tree** to make predictions.

    **Advantages**
    - Simple to understand
    - Fast to train
    - Easy to visualize

    **Limitation**
    - Can overfit the training data
    - Sensitive to the training data
    - One tree can make unstable decisions
    """)


with col2:

    st.markdown("""
    ### Random Forest

    Random Forest combines predictions from **many decision
    trees** and uses them together to make the final prediction.

    **Advantages**
    - More robust than a single tree
    - Reduces overfitting
    - More stable predictions
    - Usually provides better generalization

    **In this project**
    - 100 decision trees
    - Maximum depth: 20
    - Random state: 42
    """)


# ============================================================
# MODEL PIPELINE
# ============================================================

st.divider()

st.header("Machine Learning Pipeline")

st.markdown("""
**Vehicle Images**

↓

**Resize Images to 64 × 64**

↓

**Flatten Images into Pixel Features**

↓

**Train/Test Split**

↓

**Decision Tree Baseline**

↓

**Random Forest Optimization**

↓

**Model Evaluation**

↓

**Pickle Model**

↓

**Streamlit Deployment**
""")


# ============================================================
# TECHNOLOGIES
# ============================================================

st.divider()

st.header("Technologies Used")

tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

with tech_col1:
    st.markdown("**Python**")

with tech_col2:
    st.markdown("**Scikit-learn**")

with tech_col3:
    st.markdown("**OpenCV**")

with tech_col4:
    st.markdown("**Streamlit**")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

Vehicle Classification AI • Machine Learning Portfolio Project

Built with Python, OpenCV, Scikit-learn and Streamlit

</div>
""", unsafe_allow_html=True)