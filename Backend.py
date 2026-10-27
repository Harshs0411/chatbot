"""
Backend shim module for backward compatibility.
Allows imports like `from Backend.backend import ...` or `import Backend`
to resolve directly to the `backend` package.
"""
import sys
import backend
from backend import *

# Point Backend module in sys.modules to the backend package
sys.modules["Backend"] = backend
if hasattr(backend, "backend"):
    sys.modules["Backend.backend"] = backend.backend
