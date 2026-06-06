---
title: Iris Classification API
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Iris Classification API

Simple ML API for Iris flower classification using Random Forest.

## Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /docs` - Interactive API documentation

## Example Usage

```bash
curl -X POST https://amnaakhalid1-my-ml-app.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'iew the results and confidence scores