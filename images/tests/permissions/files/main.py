import requests
from requests.auth import HTTPBasicAuth


def request_permissions(user, password):
    api_address = 'api_from_compose'
    api_port = 8000
    return requests.get(
        url='http://{address}:{port}/permissions'.format(address=api_address, port=api_port),
        auth=HTTPBasicAuth(user, password))


def test_permissions():
    r = request_permissions("alice", "wonderland")
    assert 200 == r.status_code
    
    r = request_permissions("alice", "builder")
    assert 401 == r.status_code
