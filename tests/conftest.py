import os
import sys

# Let tests import each other (test_models reuses the tiny-model helpers).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
