from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
EXTRACT = APP.parents[1] / "data" / "dashboard" / "run_summary.json"


@pytest.mark.skipif(not EXTRACT.exists(), reason="dashboard extract not built")
def test_streamlit_app_runs_every_view():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert len(app.metric) == 4
    for level in ("item", "department", "total"):
        app.selectbox[0].set_value(level).run()
        assert not app.exception, level
    app.radio[0].set_value("backtest mean").run()
    app.radio[1].set_value("wrmsse").run()
    assert not app.exception
