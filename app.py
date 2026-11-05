import os
import sys
import runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "app", "streamlit_app.py"), run_name="__main__")
