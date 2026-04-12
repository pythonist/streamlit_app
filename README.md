# Mule Detection Client Demo

Client-ready Streamlit demo for synthetic mule-account detection across:

- data generation
- entity resolution
- feature engineering
- graph analytics and ring detection
- champion/challenger model training
- alert packaging and feedback loop

## Local Run

Create or activate the included virtual environment, then run:

```bash
streamlit run app.py
```

For the pipeline-only smoke test:

```bash
python main_pipeline.py
```

## Recommended Demo Flow

1. Open the app landing page.
2. Select a demo profile.
3. Click `Run Client Demo`.
4. Walk through `Executive Summary`, `Graph Analytics`, `Model Training`, and `Alert Engine`.

## Deployment

This repo is prepared for Streamlit Community Cloud:

- `app.py` is the entrypoint.
- `requirements.txt` contains the lean deployment dependencies.
- `.streamlit/config.toml` sets the app theme.

After pushing to GitHub, point Streamlit Community Cloud at this repo and set:

- Main file path: `app.py`
- Python dependencies: `requirements.txt`

## Notes

- The app defaults to smaller demo-scale synthetic data so it finishes fast enough for a live walkthrough.
- TensorFlow and SHAP are treated as optional extras and are disabled by default for simpler deployment.
