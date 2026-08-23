"""
Start the KneeAI backend server.
Usage: python scripts/start_backend.py
"""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["E:\\KneeAI\\backend"],
    )
