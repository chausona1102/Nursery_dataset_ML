import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'Dự đoán phân loại trẻ' in response.data.decode('utf-8')

def test_prediction(client):
    data = {
        'parents': 'usual',
        'has_nurs': 'proper',
        'form': 'complete',
        'children': 'more',
        'housing': 'convenient',
        'finance': 'convenient',
        'social': 'nonprob',
        'health': 'recommended'
    }
    response = client.post('/', data=data, follow_redirects=True)
    assert response.status_code == 200
    assert 'Dự đoán phân loại trẻ' in response.data.decode('utf-8')
    assert 'Kết quả dự đoán:' in response.data.decode('utf-8')

