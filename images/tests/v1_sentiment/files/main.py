import requests
from requests.auth import HTTPBasicAuth


api_address = "api_from_compose"
api_port = 8000


def request_v1_sentiment_predict(sentence: str):
    return requests.get(
        url='http://{address}:{port}/v1/sentiment/predict'.format(address=api_address, port=api_port),
        auth=HTTPBasicAuth("alice", "wonderland"),
        params= {"sentence": sentence})

def request_v1_sentiment_test_score():
    return requests.get(
        url='http://{address}:{port}/v1/sentiment/test_score'.format(address=api_address, port=api_port),
        auth=HTTPBasicAuth("alice", "wonderland"))

def test_v1_sentiment():
    sentence = "I love Disney"
    r = request_v1_sentiment_predict(sentence=sentence)
    assert r.status_code == 200
    
    r = request_v1_sentiment_predict(sentence=sentence)
    assert r.json().get("prediction") == "[5]"
    
    r = request_v1_sentiment_test_score()
    assert r.status_code == 200
    assert str(r.json().get("test_score"))[:6] == "0.6287"
