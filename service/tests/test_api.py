import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_known_user_gets_personalized_top3(client):
    from app.main import state
    uid = int(state['rec'].users.index[0])
    body = client.get(f'/recommend/{uid}').json()
    assert body['strategy'] == 'personalized'
    assert [x['rank'] for x in body['recommendations']] == [1, 2, 3]
    assert len({x['item_id'] for x in body['recommendations']}) == 3


def test_unknown_user_falls_back_to_popular(client):
    body = client.get('/recommend/999999999').json()
    assert body['strategy'] == 'popular_fallback' and len(body['recommendations']) == 3


def test_post_and_k(client):
    r = client.post('/recommend', json={'user_id': 1, 'k': 5})
    assert r.status_code == 200 and len(r.json()['recommendations']) == 5


@pytest.mark.parametrize('call', [
    lambda c: c.get('/recommend/abc'),
    lambda c: c.get('/recommend/-5'),
    lambda c: c.get('/recommend/1?k=0'),
    lambda c: c.get('/recommend/1?k=100'),
    lambda c: c.post('/recommend', json={'user_id': 'x'}),
    lambda c: c.post('/recommend', json={}),
    lambda c: c.post('/recommend', content=b'not json', headers={'content-type': 'application/json'}),
])
def test_bad_requests_return_422(client, call):
    assert call(client).status_code == 422


def test_metrics(client):
    client.get('/recommend/1')
    text = client.get('/metrics').text
    assert 'recsys_requests_total' in text and 'recsys_request_seconds' in text
