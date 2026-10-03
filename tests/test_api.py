import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main


def test_endpoints_filter_and_paginate(tmp_path, monkeypatch):
    forecasts = pd.DataFrame({"series_id": ["a", "a", "b"], "level": ["item", "item", "store"], "method": "mint_oos",
                              "date": pd.to_datetime(["2016-05-23", "2016-05-24", "2016-05-23"]), "forecast": [1.0, 2.0, 3.0]})
    unserved = forecasts.assign(method="bottom_up", forecast=99.0)
    pd.concat([forecasts.assign(served=True), unserved.assign(served=False)]).to_parquet(tmp_path / "forecasts.parquet")
    forecasts.assign(lower_95=0.0, upper_95=5.0).to_parquet(tmp_path / "prediction_intervals.parquet")
    pd.DataFrame({"node_id": ["x"], "allocated_quantity": [4.0]}).to_parquet(tmp_path / "allocation.parquet")
    monkeypatch.setattr(api_main, "GOLD_DIR", tmp_path)
    client = TestClient(api_main.app)

    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/get-forecast", params={"series_id": "a"}).json()
    assert body["total"] == 2 and len(body["records"]) == 2
    page = client.get("/get-forecast", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 3 and len(page["records"]) == 1
    assert client.get("/get-forecast", params={"level": "store"}).json()["total"] == 1
    assert client.get("/get-forecast", params={"method": "bottom_up"}).json()["records"][0]["forecast"] == 99.0
    assert client.get("/get-prediction-interval", params={"series_id": "b"}).json()["records"][0]["upper_95"] == 5.0
    assert client.get("/get-allocation").json()["records"][0]["allocated_quantity"] == 4.0
    assert client.get("/get-metrics", params={"table": "nope"}).status_code == 404


def _write_bundle(directory, n=30):
    import numpy as np

    rng = np.random.default_rng(0)
    forecast = np.tile(rng.uniform(1, 6, (n, 1)), (1, 28)).astype("float32")
    np.savez_compressed(
        directory / "network.npz", forecast=forecast, history=rng.poisson(3, (n, 28)).astype("float32"), scale=np.maximum(forecast[:, 0], 1),
        current_safety=np.full(n, 3.0, dtype="float32"), safety_by_service=np.tile(np.linspace(1, 6, 8, dtype="float32"), (n, 1)),
        review=np.full(n, 3, dtype="int8"), price=np.ones(n, dtype="float32"), score_paths=rng.normal(0, 1, (300, 28)).astype("float16"),
        dept_ids=np.array(["FOODS_1"] * n), cat_ids=np.array(["FOODS"] * n),
        store_ids=np.array(["CA_1", "CA_2"] * (n // 2)), item_ids=np.repeat(np.array([f"FOODS_1_{i:03d}" for i in range(n // 2)]), 2),
    )


def test_api_boots_on_dashboard_extract_alone_and_serves_twin(tmp_path, monkeypatch):
    pd.DataFrame({"fold": ["holdout"], "metric": ["fill_rate"], "predicted": [0.97]}).to_parquet(tmp_path / "twin_validation.parquet")
    twin_dir = tmp_path / "twin_inputs"
    twin_dir.mkdir()
    _write_bundle(twin_dir)
    monkeypatch.setattr(api_main, "GOLD_DIR", tmp_path)
    monkeypatch.setattr(api_main, "TWIN_DIR", twin_dir)
    api_main.load_bundle.cache_clear()
    api_main.cached_simulation.cache_clear()
    api_main._hits.clear()
    client = TestClient(api_main.app)

    assert client.get("/health").json()["twin_stores"] == ["CA_1", "CA_2"]
    assert client.get("/twin/tables/twin_validation").json()["records"][0]["predicted"] == 0.97
    assert client.get("/twin/tables/not_a_table").status_code == 404
    spike = {"store_id": "CA_1", "reps": 6, "scenario": {"demand_scale": 1.8, "category": "FOODS", "days": [5, 12]}}
    body = client.post("/twin/simulate", json=spike).json()
    assert body["scenario"]["kpis"]["all"]["lost_sales_value"]["mean"] >= body["baseline"]["kpis"]["all"]["lost_sales_value"]["mean"]
    assert set(body["baseline"]["kpis"]) == {"all", "CA_1", "CA_2"}
    assert client.post("/twin/simulate", json=spike).json() == body  # cached and deterministic
    assert client.post("/twin/simulate", json={"store_id": "CA_9"}).status_code == 404


def test_twin_request_caps_return_422(tmp_path, monkeypatch):
    client = TestClient(api_main.app)
    assert client.post("/twin/simulate", json={"store_id": "CA_1", "reps": 5000}).status_code == 422
    assert client.post("/twin/simulate", json={"store_id": "CA_1", "scenario": {"demand_scale": 50}}).status_code == 422
    assert client.post("/twin/simulate", json={"store_id": "../etc"}).status_code == 422
    assert client.post("/twin/simulate", json={"store_id": "CA_1", "service": 0.5}).status_code in (404, 422)


def test_rate_limit_and_cors_and_proxy_ip(tmp_path, monkeypatch):
    twin_dir = tmp_path / "twin_inputs"
    twin_dir.mkdir()
    _write_bundle(twin_dir)
    monkeypatch.setattr(api_main, "TWIN_DIR", twin_dir)
    monkeypatch.setattr(api_main, "RATE_LIMIT", 3)
    api_main.load_bundle.cache_clear()
    api_main.cached_simulation.cache_clear()
    api_main._hits.clear()
    client = TestClient(api_main.app)
    statuses = [client.post("/twin/simulate", json={"store_id": "CA_1", "reps": 2, "seed": seed}).status_code for seed in range(5)]
    assert statuses == [200, 200, 200, 429, 429]
    allowed = client.options("/twin/simulate", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    blocked = client.options("/twin/simulate", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in blocked.headers

    class Fake:
        headers = {"x-forwarded-for": "6.6.6.6, 1.2.3.4"}
        client = type("C", (), {"host": "10.0.0.1"})()

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    assert api_main.client_ip(Fake()) == "1.2.3.4"  # a client cannot spoof an earlier entry
    monkeypatch.delenv("TRUSTED_PROXY_HOPS")
    assert api_main.client_ip(Fake()) == "10.0.0.1"
