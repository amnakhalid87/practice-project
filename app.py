from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Initialize FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="Simple ML API for Iris flower classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)


# Request model
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Response model
class IrisResponse(BaseModel):
    species: str
    confidence: float
    probabilities: dict


# Load or train model
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        iris = load_iris()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(iris.data, iris.target)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model


model = load_model()
class_names = ['setosa', 'versicolor', 'virginica']


# Routes
@app.get("/")
def root():
    return {"message": "Iris Classification API", "endpoints": ["/predict", "/health"]}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    # Prepare input
    input_data = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]

    # Predict
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(max(probabilities) * 100)

    return IrisResponse(
        species=class_names[prediction],
        confidence=round(confidence, 2),
        probabilities={
            class_names[0]: round(probabilities[0] * 100, 2),
            class_names[1]: round(probabilities[1] * 100, 2),
            class_names[2]: round(probabilities[2] * 100, 2)
        }
    )