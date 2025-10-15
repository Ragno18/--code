import sys
import os
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_get_stocks_no_params(client):
    """
    Test that the /api/stocks endpoint returns a successful response
    and a JSON payload when no query parameters are provided.
    """
    response = client.get('/api/stocks')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) <= 5
    if data:
        for stock in data:
            assert 'symbol' in stock
            assert 'price' in stock