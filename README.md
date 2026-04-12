# PwC Mule Model Studio

Streamlit workspace for synthetic mule-account analysis across:

- data generation
- entity resolution
- feature engineering
- graph analytics and ring detection
- champion and challenger model training
- alert packaging and feedback

## Local Run

```bash
streamlit run app.py
```

## Pipeline Smoke Test

```bash
python main_pipeline.py
```

## Workflow

1. Choose a run profile.
2. Click `Run Pipeline`.
3. Review the overview, graph analytics, model training, and alert pages.

## Deployment

- Entry point: `app.py`
- Dependencies: `requirements.txt`
- Theme config: `.streamlit/config.toml`

## Notes

- The default profiles use smaller synthetic samples so the graph and model steps stay responsive.
- TensorFlow and SHAP are disabled by default to keep deployment lightweight and stable.
