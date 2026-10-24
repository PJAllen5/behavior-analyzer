"""Kept so `streamlit run app.py` still works. The app itself is behavetrack/app.py;
the usual way to start it is `behavetrack app` or the Start Behavior Tracker launcher."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "behavetrack" / "app.py"), run_name="__main__")
