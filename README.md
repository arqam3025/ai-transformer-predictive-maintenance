# AI Transformer Predictive Maintenance

An independent engineering and machine-learning project by **Raja Arqam Abdullah** for transformer fault diagnosis, dissolved gas analysis (DGA), condition trending, asset health assessment and maintenance prioritisation.

> **Status:** Active development. This repository is a new implementation and is not a fork.

## Project objective

The goal is to build an end-to-end, explainable condition-monitoring pipeline:

```text
DGA measurements
      ↓
data validation & feature engineering
      ↓
ML fault classification
      ↓
model evaluation & explainability
      ↓
longitudinal deterioration analysis
      ↓
asset health / risk assessment
      ↓
fleet maintenance prioritisation
      ↓
engineering dashboard
```

The project is designed as decision-support software. It will not claim that custom health scores or thresholds are IEC/IEEE standard limits unless they are directly implemented and validated from an appropriate source.

## Current stage — DGA foundation

The first implementation stage provides a typed five-gas DGA sample model for H₂, CH₄, C₂H₆, C₂H₄ and C₂H₂; validation for missing, negative and non-finite measurements; transparent feature engineering; total-gas, ratio and normalized gas-fraction features; and automated unit tests.

Gas ratios are used as **model features** at this stage; they are not presented as a standalone standards-based diagnosis.

## Planned stages

1. DGA data model and feature engineering — **in progress**
2. Dataset ingestion and reproducible train/test pipeline
3. Baseline ML fault classifier and evaluation
4. Explainability and uncertainty analysis
5. Longitudinal DGA deterioration modelling
6. Transformer health and risk model
7. Fleet maintenance prioritisation
8. Streamlit engineering dashboard
9. CI, documentation and research evaluation

## Development

The active implementation is developed on the `develop` branch before reviewed changes are merged to `main`.

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
```

## Responsible engineering use

This is a research and portfolio project, not a certified protection, diagnostic or maintenance system. Model outputs must be validated on appropriate transformer data before operational use. Final maintenance decisions require applicable standards, test evidence and qualified engineering judgement.

## Author

**Raja Arqam Abdullah**  
MSc Artificial Intelligence | Electrical Engineering background  
GitHub: [arqam3025](https://github.com/arqam3025)

## License

MIT License. See [LICENSE](LICENSE).
