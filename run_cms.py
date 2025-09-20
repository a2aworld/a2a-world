#!/usr/bin/env python3
"""
Script to run the Galactic Storybook CMS
"""
import uvicorn
from backend.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
