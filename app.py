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
# CSS
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

p, li {
    color: #374151 !important;
    font-size: 16px;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #dfe5ee;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    color: #172033 !important;
}

[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #dce3ed;
}

hr {
    border-color: #dce2eb;
}

.footer {
    text-align: center;
    color: #64748b;
    padding-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

MODEL_PATH = "vehicle_random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        "vehicle_random_forest_model.pkl was not found. "
        "Please place the pickle file in the same folder as app.py."
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
# ACTUAL MODEL RESULTS
# ============================================================

decision_tree_accuracy = 43.38
random_forest_accuracy = 62.16

accuracy_improvement = (
    random_forest_accuracy -
    decision_tree_accuracy
)

relative_improvement = (
    accuracy_improvement /
    decision_tree_accuracy
) * 100


# ============================================================
# HEADER
# ============================================================

st.title("VehicleVision AI")

st.subheader(
    "Vehicle Image Classification with Machine Learning"
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
        "Decision Tree",
        "43.38%"
    )

with col4:
    st.metric(
        "Random Forest",
        "62.16%"
    )


st.write("")


# ============================================================
# MODEL IMPROVEMENT
# ============================================================

st.header("Model Improvement")

st.write(
    "The Decision Tree was first developed as the baseline model. "
    "Random Forest was then applied as an ensemble method, "
    "combining multiple decision trees to produce more robust predictions."
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Decision Tree Accuracy",
        "43.38%"
    )

with col2:

    st.metric(
        "Random Forest Accuracy",
        "62.16%"
    )

with col3:

    st.metric(
        "Accuracy Gain",
        f"+{accuracy_improvement:.2f}%"
    )


st.write("")

comparison_df = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy (%)": [
        decision_tree_accuracy,
        random_forest_accuracy
    ]
})

st.bar_chart(
    comparison_df.set_index("Model")
)

st.success(
    f"Random Forest increased test accuracy from "
    f"{decision_tree_accuracy:.2f}% to "
    f"{random_forest_accuracy:.2f}%, "
    f"a gain of {accuracy_improvement:.2f} percentage points."
)


st.caption(
    f"Relative improvement over the Decision Tree: "
    f"{relative_improvement:.1f}%."
)


# ============================================================
# IMPORTANT ACCURACY EXPLANATION
# ============================================================

st.info(
    "Model Accuracy (62.16%) is measured across the complete "
    "test dataset. It is different from the confidence shown "
    "for an individual uploaded image."
)


# ============================================================
# WHY RANDOM FOREST?
# ============================================================

st.header("Why Random Forest?")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Decision Tree")

    st.write(
        "The Decision Tree uses one tree to make predictions. "
        "It provided a baseline accuracy of 43.38%."
    )

    st.write(
        "A single tree can be sensitive to the training data "
        "and may produce less stable predictions."
    )


with col2:

    st.subheader("Random Forest")

    st.write(
        "Random Forest combines many decision trees and "
        "aggregates their predictions."
    )

    st.write(
        "This ensemble approach produced a test accuracy "
        "of 62.16%, improving the baseline by 18.78 percentage points."
    )


st.divider()


# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

st.header("Try the Model")

st.write(
    "Upload a vehicle image to see the prediction made by "
    "the trained Random Forest model."
)

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "The uploaded image could not be read."
        )

    else:

        # ----------------------------------------------------
        # PREPROCESS IMAGE
        # ----------------------------------------------------

        resized_image = cv2.resize(
            image,
            (64, 64)
        )

        flattened_image = (
            resized_image
            .flatten()
            .reshape(1, -1)
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            flattened_image
        )

        predicted_index = int(
            prediction[0]
        )

        predicted_class = classes[
            predicted_index
        ]


        # ----------------------------------------------------
        # INDIVIDUAL IMAGE CONFIDENCE
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                flattened_image
            )[0]

            image_confidence = (
                np.max(probabilities) * 100
            )

        else:

            probabilities = None
            image_confidence = 0


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Uploaded Vehicle Image",
                use_container_width=True
            )


        with col2:

            st.subheader("AI Prediction")

            st.success(
                f"Predicted Vehicle: {predicted_class}"
            )

            st.metric(
                "Confidence for This Image",
                f"{image_confidence:.2f}%"
            )

            st.caption(
                "This confidence score applies only to this "
                "uploaded image. It is NOT the overall model accuracy."
            )


        # ----------------------------------------------------
        # PROBABILITY BREAKDOWN
        # ----------------------------------------------------

        if probabilities is not None:

            st.subheader(
                "Prediction Probability"
            )

            probability_df = pd.DataFrame({
                "Vehicle": classes,
                "Probability (%)": (
                    probabilities * 100
                )
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


# ============================================================
# MACHINE LEARNING PIPELINE
# ============================================================

st.divider()

st.header("Machine Learning Pipeline")

pipeline_df = pd.DataFrame({
    "Stage": [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08"
    ],
    "Process": [
        "Vehicle Image Dataset",
        "Resize Images to 64 × 64",
        "Flatten Images into Pixel Features",
        "Train/Test Split",
        "Decision Tree Baseline",
        "Random Forest Training",
        "Model Evaluation",
        "Pickle + Streamlit Deployment"
    ]
})

st.dataframe(
    pipeline_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VEHICLE CLASSES
# ============================================================

st.header("Supported Vehicle Categories")

category_df = pd.DataFrame({
    "Label": range(7),
    "Category": classes
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
    "Python  •  OpenCV  •  NumPy  •  Pandas  •  "
    "Scikit-learn  •  Random Forest  •  Pickle  •  Streamlit"
)


# ============================================================
# PROJECT RESULT
# ============================================================

st.divider()

st.header("Final Result")

st.write(
    f"The initial Decision Tree achieved an accuracy of "
    f"{decision_tree_accuracy:.2f}%. After moving to Random Forest, "
    f"the test accuracy increased to {random_forest_accuracy:.2f}%. "
    f"This represents an improvement of "
    f"{accuracy_improvement:.2f} percentage points."
)


st.success(
    "Decision Tree: 43.38%  →  Random Forest: 62.16%"
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