"""Optional Prophet adapter with M5 calendar events as holidays."""
from __future__ import annotations

import logging

import pandas as pd


def calendar_holidays(calendar: pd.DataFrame) -> pd.DataFrame:
    """Turn M5 ``event_name_1``/``event_name_2`` columns into a Prophet holidays frame."""
    frames = []
    for column in ("event_name_1", "event_name_2"):
        if column in calendar:
            events = calendar.loc[calendar[column].notna(), ["date", column]]
            frames.append(events.rename(columns={"date": "ds", column: "holiday"}))
    if not frames:
        return pd.DataFrame(columns=["ds", "holiday"])
    holidays = pd.concat(frames, ignore_index=True)
    holidays["ds"] = pd.to_datetime(holidays["ds"])
    holidays["holiday"] = holidays["holiday"].astype(str)
    return holidays.drop_duplicates()


def prophet_forecast(history: pd.DataFrame, horizon: int, holidays: pd.DataFrame | None = None, interval_width: float = 0.95) -> pd.DataFrame:
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, holidays=holidays, interval_width=interval_width)
    train = history.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]].copy()
    model.fit(train)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    output = model.predict(future).tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    output = output.rename(columns={"ds": "date", "yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"})
    output[["forecast", "lower", "upper"]] = output[["forecast", "lower", "upper"]].clip(lower=0)
    return output.reset_index(drop=True)
