"""
backend/mcp_server.py: Package-level entrypoint forwarding to root mcp_server.py.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mcp_server import main, mcp

if __name__ == "__main__":
    main()
