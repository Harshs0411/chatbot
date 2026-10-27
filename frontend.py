"""
Streamlit Root Entrypoint for LangGraph Conversational Agent & RAG.
Enables running:
    streamlit run frontend.py
directly from the project root directory, in addition to:
    streamlit run frontend/frontend.py
"""

import os
import sys
import runpy

# Ensure project root directory is on sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Path to the actual frontend application
FRONTEND_SCRIPT = os.path.join(ROOT_DIR, "frontend", "frontend.py")

# Execute the application in the current module's namespace or via run_path
runpy.run_path(FRONTEND_SCRIPT, run_name="__main__")
