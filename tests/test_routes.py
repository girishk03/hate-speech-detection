def test_homepage_loads(client):
    response = client.get("/")

    assert response.status_code == 200


def test_prediction_endpoint_accepts_valid_request(client, combined_module, monkeypatch):
    class DeferredThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

    monkeypatch.setattr(combined_module.threading, "Thread", DeferredThread)
    response = client.post("/analyze", json={"url": "https://www.youtube.com/watch?v=example"})

    assert response.status_code == 200
    assert response.get_json()["job_id"] in combined_module.analysis_jobs


def test_prediction_endpoint_rejects_empty_url(client):
    response = client.post("/analyze", json={"url": ""})

    assert response.status_code == 400
    assert response.get_json() == {"success": False, "error": "A YouTube URL is required"}


def test_prediction_endpoint_rejects_invalid_payload(client):
    response = client.post("/analyze", data="not-json", content_type="text/plain")

    assert response.status_code == 400
    assert response.get_json()["success"] is False
