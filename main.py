# main.py
import subprocess
from src.logger import logging
from src.exception import CustomException

def run_module(module_name):
    logging.info("Running module: %s", module_name)
    subprocess.run(["python", "-m", module_name], check=True)

def main():
    logging.info("=== Full ANN Pipeline Started ===")
    try:
        run_module("src.run_preprocessing")
        run_module("src.run_training")
        run_module("src.run_evaluate")   # if present
        logging.info("=== Full ANN Pipeline Completed Successfully ===")
    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Pipeline execution failed: %s", e)
