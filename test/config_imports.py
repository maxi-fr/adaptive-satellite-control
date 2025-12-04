import sys
import os

try:
    # Normal case (running pytest)
    HERE = os.path.dirname(__file__)
except NameError:
    # Fallback (Jupyter, IPython)
    HERE = os.getcwd()

src_path = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, src_path)
