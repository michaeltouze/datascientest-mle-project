from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
from typing import Optional, Tuple, Union, Any, Dict
import pickle
import secrets
import json
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

app = FastAPI(title='My API')
security = HTTPBasic()

def load_credentials():
    with open("./permissions/credentials.json", "r") as f:
        credentials_db = json.load(f)
    return credentials_db


def check_user_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    credentials_db = load_credentials()
    for user, pwd in credentials_db.items():
        correct_username = secrets.compare_digest(credentials.username, user)
        correct_password = secrets.compare_digest(credentials.password, base64.b64decode(pwd.encode()).decode("ascii"))
        if correct_username & correct_password:
            return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Basic"})


def load_model(
    version: str, 
    location: Optional[str] = None) -> Tuple[Union[LogisticRegression,RandomForestClassifier], CountVectorizer]:

    with open("./data/model_{version}.pkl".format(version=version), "rb") as f:
        model = pickle.load(f)
    with open("./data/param_{version}.pkl".format(version=version), "rb") as f:
        count_vectorizer = pickle.load(f)

    if version == "v1":
        return model, count_vectorizer
    elif version == "v2":
        if location:
            if location in model.keys():
                return model[location], count_vectorizer[location]
            else:
                raise HTTPException(status_code=404, detail="Disney Parc location not found")
        else:
            raise HTTPException(status_code=404, detail="Disney Parc location is required")


def load_performance(version: str, location: Optional[str] = None) -> float:
    with open("./data/performance_{version}.pkl".format(version=version), "rb") as f:
        score = pickle.load(f)

    if version == "v1":
        return score
    if version == "v2":
        if location:
            if location in score.keys():
                return score[location]
            else:
                raise HTTPException(status_code=404, detail="Disney Parc location not found")
        else:
            return score


def get_model_output(
    cv: CountVectorizer, 
    model: Union[LogisticRegression,RandomForestClassifier], 
    sentence: str) -> Dict[str, str]:

    X = np.array([sentence])
    X_transform = cv.transform(X)
    y_pred = model.predict(X_transform)
    y_pred_proba = model.predict_proba(X_transform)
    out = dict(prediction=str(y_pred))
    return out

@app.get("/status")
async def get_status():
    """Returns 1 if API is up
    """
    return {"status": 1}


@app.get("/permissions")
async def get_user_authorization(username: str = Depends(check_user_credentials)):
    """Returns user authorization for using API
    """
    return {"authorized": True}


@app.get("/v1/sentiment/predict")
async def get_v1_sentiment_predict(sentence: str, username: str = Depends(check_user_credentials)):
    """Returns predicted note of an input sentence based a Logistic Regression trained model
    """
    model, count_vectorizer = load_model(version="v1")
    out = get_model_output(cv=count_vectorizer, model=model, sentence=sentence)
    return out


@app.get("/v1/sentiment/test_score")
async def get_v1_sentiment_test_score(username: str = Depends(check_user_credentials)):
    """Return v1 performance score on test sample
    """
    score = load_performance(version="v1")
    return {"test_score": score}


@app.get("/v2/sentiment/predict")
async def get_v2_sentiment_predict(sentence: str, location: str, username: str = Depends(check_user_credentials)):
    """Returns predicted note of an input sentence based a Random Forest Classifier trained model, 
    split on each Disney Parc
    - Disneyland_HongKong
    - Disneyland_California
    - Disneyland_Paris
    """
    model, count_vectorizer = load_model(version="v2", location=location)
    out = get_model_output(cv=count_vectorizer, model=model, sentence=sentence)
    return out


@app.get("/v2/sentiment/test_score")
async def get_v1_sentiment_test_score(location: Optional[str] = None, username: str = Depends(check_user_credentials)):
    """Return v2 performance scores on test sample
    """
    score = load_performance(version="v2", location=location)
    return {"test_score": score}
