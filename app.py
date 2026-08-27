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
    layout="wide"
)

# ============================================================
# SIMPLE PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f7f9fc;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */
h1 {
    color: #172033 !important;
    font-weight: 800 !important;
}

h2 {
    color: #172033 !important;
    font-weight: 750 !important;
}

h3 {
    color: #26344d !important;
}

/* Normal text */
p, li, label {
    color: #374151 !important;
    font-size: 16px;
}

/* Metric styling */
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e1e6ef;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    color: #172033 !important;
}

/* Divider */
hr {
    border-color: #dce2eb;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #dce2eb;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "vehicle_random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model file 'vehicle_random_forest_model.pkl' was not found. "
        "Place the pickle file in the same folder as app.py."
    )

    st.stop()


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ============================================================
# VEHICLE CLASSES
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
# ACCURACY
# ============================================================

# Your confirmed Decision Tree accuracy
decision_tree_accuracy = 43.38

# CHANGE THIS AFTER RUNNING YOUR RANDOM FOREST CODE
# Example: if RF accuracy = 58.75%, write 58.75
random_forest_accuracy = 0.0


# ============================================================
# HEADER
# ============================================================

st.title("VehicleVision AI")

st.subheader(
    "Vehicle Image Classification using Machine Learning"
)

st.write(
    "An end-to-end computer vision project that classifies "
    "vehicle images into seven categories using a Random Forest model."
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.header("Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Vehicle Classes",
        "7"
    )

with col2:
    st.metric(
        "Dataset Images",
        "5,590"
    )

with col3:
    st.metric(
        "Image Size",
        "64 × 64"
    )

with col4:
    st.metric(
        "Decision Tree",
        "43.38%"
    )


st.write("")


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("Model Performance")

st.write(
    "The Decision Tree was first used as a baseline model. "
    "Random Forest was then applied as an ensemble approach "
    "to improve the model's generalization and prediction stability."
)

if random_forest_accuracy > 0:

    improvement = (
        random_forest_accuracy -
        decision_tree_accuracy
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
            "Improvement",
            f"+{improvement:.2f}%"
        )

    comparison = pd.DataFrame({
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
        comparison.set_index("Model")
    )

    st.success(
        f"Random Forest improved the accuracy by "
        f"{improvement:.2f} percentage points "
        f"({relative_improvement:.1f}% relative improvement)."
    )

else:

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Decision Tree Accuracy",
            "43.38%"
        )

    with col2:
        st.metric(
            "Random Forest Accuracy",
            "Run Model"
        )

    st.info(
        "After running the Random Forest training code, "
        "enter its accuracy in the code above to complete "
        "the comparison."
    )


# ============================================================
# WHY RANDOM FOREST?
# ============================================================

st.header("Why Random Forest?")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Decision Tree")

    st.write(
        "A Decision Tree makes predictions using a single "
        "tree. It is simple and easy to interpret, but its "
        "performance can be sensitive to the training data."
    )

    st.write("**Baseline Accuracy: 43.38%**")


with col2:

    st.subheader("Random Forest")

    st.write(
        "Random Forest combines multiple Decision Trees "
        "and aggregates their predictions. This makes the "
        "model more robust and generally reduces the risk "
        "of relying on one unstable tree."
    )

    st.write("**Final Model: Random Forest**")


st.divider()


# ============================================================
# IMAGE PREDICTION
# ============================================================

st.header("Try VehicleVision AI")

st.write(
    "Upload a vehicle image and the trained Random Forest "
    "model will predict its category."
)

uploaded_file = st.file_uploader(
    "Choose a vehicle image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error("Unable to read the uploaded image.")

    else:

        # Resize exactly as training
        resized_image = cv2.resize(
            image,
            (64, 64)
        )

        # Flatten exactly as training
        flattened_image = resized_image.flatten().reshape(
            1, -1
        )

        # Prediction
        prediction = model.predict(
            flattened_image
        )

        predicted_index = int(
            prediction[0]
        )

        predicted_class = classes[
            predicted_index
        ]

        # Probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                flattened_image
            )[0]

            confidence = (
                np.max(probabilities) * 100
            )

        else:

            probabilities = None
            confidence = 0


        # ----------------------------------------------------
        # DISPLAY IMAGE + RESULT
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Uploaded Vehicle",
                use_container_width=True
            )

        with col2:

            st.subheader("Prediction")

            st.success(
                f"Vehicle: {predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        # ----------------------------------------------------
        # PROBABILITY BREAKDOWN
        # ----------------------------------------------------

        if probabilities is not None:

            st.subheader(
                "Prediction Probability by Class"
            )

            probability_df = pd.DataFrame({
                "Vehicle": classes,
                "Probability (%)":
                    probabilities * 100
            })

            probability_df = (
                probability_df
                .sort_values(
                    "Probability (%)",
                    ascending=False
                )
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                probability_df.set_index(
                    "Vehicle"
                )
            )


# ============================================================
# MACHINE LEARNING PIPELINE
# ============================================================

st.divider()

st.header("Machine Learning Pipeline")

pipeline = pd.DataFrame({
    "Step": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7"
    ],
    "Process": [
        "Vehicle Image Dataset",
        "Resize Images to 64 × 64",
        "Convert Images to Pixel Features",
        "Train/Test Split",
        "Decision Tree Baseline",
        "Random Forest Training",
        "Pickle Model + Streamlit Deployment"
    ]
})

st.dataframe(
    pipeline,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATASET CLASSES
# ============================================================

st.header("Vehicle Categories")

category_df = pd.DataFrame({
    "Label": range(7),
    "Vehicle Category": classes
})

st.dataframe(
    category_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.header("Technology Stack")

st.write(
    "**Python**  •  "
    "**OpenCV**  •  "
    "**NumPy**  •  "
    "**Pandas**  •  "
    "**Scikit-learn**  •  "
    "**Random Forest**  •  "
    "**Pickle**  •  "
    "**Streamlit**"
)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.divider()

st.header("Project Summary")

st.write(
    "This project demonstrates how a traditional Machine "
    "Learning approach can be applied to image classification. "
    "A Decision Tree established the initial baseline with "
    "43.38% accuracy. Random Forest was subsequently used "
    "to combine multiple decision trees and create a more "
    "robust classification model."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        VehicleVision AI | Machine Learning Portfolio Project
    </div>
    """,
    unsafe_allow_html=True
)