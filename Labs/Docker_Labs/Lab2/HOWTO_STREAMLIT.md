Run the Streamlit dashboard for Lab2 (Wine classifier)

This project includes a Streamlit dashboard (`src/Dashboard_wine.py`) which calls the local Flask prediction server (http://localhost:4000/predict).

Quick-start (recommended for development):

1. Activate the conda environment created earlier (mlopslabs):

   conda activate mlopslabs

2. Run the helper script from the repository root:

   bash Labs/Docker_Labs/Lab2/run_streamlit.sh

3. Open the Streamlit UI at:

   http://localhost:8501

Notes:
- Ensure the Flask prediction server is running (the Docker image `lab2-wine-serve` or local `python src/main.py`).
- Place class images `class0.jpeg`, `class1.jpeg`, and `class2.jpeg` in `Labs/Docker_Labs/Lab2/src/statics` if you want images to be displayed in the dashboard.
- If you prefer a containerized dashboard, create a `dockerfile.streamlit` and build it; this repo does not include that by default.
