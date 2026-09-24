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
