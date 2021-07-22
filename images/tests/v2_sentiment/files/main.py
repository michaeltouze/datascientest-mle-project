import requests
from requests.auth import HTTPBasicAuth


api_address = "api_from_compose"
api_port = 8000


def request_v2_sentiment_predict(sentence: str, location: str):
    return requests.get(
        url='http://{address}:{port}/v2/sentiment/predict'.format(address=api_address, port=api_port),
        auth=HTTPBasicAuth("alice", "wonderland"),
        params= {"sentence": sentence, "location": location})


def request_v2_sentiment_test_score():
    return requests.get(
        url='http://{address}:{port}/v2/sentiment/test_score'.format(address=api_address, port=api_port),
        auth=HTTPBasicAuth("alice", "wonderland"))


def test_v2_sentiment():
    sentence = "I love Disney"
    r = request_v2_sentiment_predict(sentence=sentence, location="Disneyland_Paris")
    assert r.status_code == 200
    
    r = request_v2_sentiment_predict(sentence=sentence, location="Disneyland")
    assert r.status_code == 404
    
    r = request_v2_sentiment_predict(sentence=sentence, location="Disneyland_Paris")
    assert r.json().get("prediction") == "[5]"
    
    r = request_v2_sentiment_test_score()
    assert r.status_code == 200
    assert str(r.json().get("test_score").get("Disneyland_Paris"))[:6] == "0.4501"
