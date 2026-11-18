# PathSnap: Distributed Map-Matching and Route Reconstruction

PES University — Machine Learning Hackathon

## Objective

Reconstruct realistic vehicle routes from noisy GPS observations
using machine learning and HMM/Viterbi decoding.

## Dataset

The dataset contains:
- 80 training trips
- 10 student-testing trips
- 10 evaluator-testing trips reserved for final evaluation

Each trip may contain:
- .track: GPS longitude, latitude and timestamp
- .nodes: road-network node coordinates
- .arcs: road-network connections
- .route: reference road-segment sequence

The raw dataset is stored locally in:
Student folder/train/
Student folder/student_test/

## Project Components

- Dataset exploration and preprocessing
- Candidate road-segment generation
- Feature engineering
- Five-model experimentation and comparison
- Bayesian hyperparameter optimisation
- HMM/Viterbi route reconstruction
- Folium geographical visualisation
- FastAPI backend
- Streamlit frontend
- Technical report and experiment records

## Setup

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

## Development

Source code is located in src/pathsnap/.

Executable scripts are located in scripts/.

## Evaluation

The backend runs on Laptop 1.
The frontend runs on Laptop 2.

Evaluator-testing ground truth must remain server-side and
must never be used to generate predictions.

## Status

Initial project structure created.
Implementation in progress.
