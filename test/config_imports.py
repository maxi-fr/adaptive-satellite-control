import sys
import os

try:
    # Normal case (running pytest)
    PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
except NameError:
    # Fallback (Jupyter, IPython)
    PROJECT_DIR = os.path.dirname(os.getcwd())

src_path = os.path.abspath(os.path.join(PROJECT_DIR, "src"))
sys.path.insert(0, src_path)

# print(PROJECT_DIR)