# ──────────────────────────────────────────────────────────────────────────────
# app.py – Decay-Heat Explorer (Streamlit version)
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

appdir = os.getcwd()  # Gets current notebook directory
root = os.path.abspath(os.path.join(appdir))
sys.path.insert(0, root)

from data_loader import DataLoader
from model_trainer import ModelTrainer
from visualizer import Visualizer

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── UI CONFIG & THEME ─────────────────────────────
st.set_page_config(
    page_title="DecayHeatML",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Soft gradient background for a modern feel */
      .stApp {background: linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);}
      /* Hide Streamlit footer */
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 DecayHeatML")

# ────────────────────── CACHED DATA + MODEL PREP (EXPENSIVE) ─────────────────
DATA_FILE = root + "/data/FLi_FTh4_FU4.pickle"   # adjust if your path differs

@st.cache_resource(show_spinner="Loading data & training models…")
def prepare_trainer() -> ModelTrainer:
    loader = DataLoader(DATA_FILE)
    _, df = loader.load_and_preprocess()
    trainer = ModelTrainer(df)
    trainer.train_and_evaluate()          # fills trainer.models & trainer.results
    return trainer

try:
    trainer = prepare_trainer()
except FileNotFoundError:
    st.error(f"Cannot find `{DATA_FILE}`.  "
             "Check the path or edit `DATA_FILE` at the top of *app.py*.")
    st.stop()

viz = Visualizer(trainer)                 # re-uses your existing plotting utils
MODEL_NAMES = list(trainer.models.keys())

# ─────────────────────────── SIDEBAR CONTROLS ────────────────────────────────
st.sidebar.header("⚙️ Controls")

model_name = st.sidebar.selectbox("Model", MODEL_NAMES, index=0)

power_density     = st.sidebar.slider(
    "Power Density (W cm⁻³)",      min_value=1.0,  max_value=100.0, value=10.0, step=0.1
)
air_ingress       = st.sidebar.slider(
    "Air Ingress (fraction)",      min_value=0.0,  max_value=0.1,  value=0.0,   step=1e-4,
    format="%.4f"
)
humidity_content  = st.sidebar.slider(
    "Humidity Content (fraction)", min_value=0.0,  max_value=0.1,  value=0.0,   step=1e-4,
    format="%.4f"
)
polynomial_order  = st.sidebar.slider("Polynomial Order", 1, 10, 2, 1)
num_regions       = st.sidebar.slider("Number of Regions", 1, 10, 1, 1)

# ────────────────────────── MAIN PLOT AREA ───────────────────────────────────
with st.spinner("Generating prediction plot…"):
    # The original helper draws directly with Matplotlib
    viz._plot_predictions(
        model_name,
        power_density,
        air_ingress,
        humidity_content,
        polynomial_order,
        num_regions,
    )
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)

# ───────────────────────── OPTIONAL EVAL REPORT ──────────────────────────────
with st.expander("🔎 Show detailed model-evaluation report"):
    REPORT_FILE = "model_report.png"
    if not os.path.exists(REPORT_FILE):
        viz.create_model_evaluation_report(save_path=REPORT_FILE)
    st.image(REPORT_FILE, use_container_width=True)

st.caption(
    "Built with Streamlit · models auto-trained with scikit-learn · "
    "visuals based on your original Jupyter workflow."
)
