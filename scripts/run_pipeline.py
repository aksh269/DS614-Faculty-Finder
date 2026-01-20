import logging
import sys
import threading
import uvicorn


import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure the config directory is correctly structured and accessible
from config.settings import RAW_DATA_PATH, CLEANED_DATA_PATH, DATABASE_PATH, ENABLE_SCRAPING, ENABLE_TRANSFORMATION, ENABLE_DATABASE_INSERTION
from ingestion.run_scrapper import run_scraping_pipeline  
from transformation.transform_pipeline import transform_file
from storage.database_insertion import Data_insertion
from fastapi import FastAPI


def start_api():
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    # On Windows, this guard is required for multiprocessing
    # import multiprocessing
    # multiprocessing.freeze_support()  # optional, but safe on Windows

    start_api()
