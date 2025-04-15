#!/usr/bin/env python
#!/usr/bin/env python
"""
workflow.py

Automation Workflow for VERA Forecasting
-----------------------------------------
This script performs the following:
  1. Checks that all required Python packages are installed and installs any that are missing.
  2. Verifies that R is installed on the system.
  3. Ensures that the R helper package "vera4castHelpers" is installed.
  4. Runs the core forecasting code (main.py) to generate forecasts.
  5. Validates and submits the forecast output using R helper functions from vera4castHelpers.

"""

import os
import sys
import subprocess

import main  # Import the core forecasting module
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, isinstalled
import sys
import subprocess
import sys
import subprocess

import sys
import subprocess

def check_python_dependencies():
    print("Checking Python dependencies...")
    required = ["numpy", "pandas", "matplotlib", "sklearn", "tensorflow", "rpy2"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("Missing Python packages detected:", missing)
        print("Attempting to install missing packages...")
        for pkg in missing:
            try:
                print(f"Installing {pkg}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except subprocess.CalledProcessError:
                print(f"Failed to install {pkg}. Please install it manually and retry.")
                sys.exit(1)
        # Re-check after installation
        for pkg in missing:
            try:
                __import__(pkg)
            except ImportError:
                print(f"Package {pkg} still cannot be imported after attempted installation. Aborting.")
                sys.exit(1)
        print("Successfully installed all missing Python packages.")
    else:
        print("All required Python packages are already installed.")

def check_R_installation():
    print("Checking if R is installed...")
    try:
        # This runs the "Rscript --version" command to check if R is in the PATH.
        subprocess.run(["Rscript", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("R is installed.")
    except Exception:
        print("R is not installed on this system. Please install R (for example: sudo apt-get install r-base) and try again.")
        sys.exit(1)
def ensure_r_helpers():
    print("Ensuring R helper package 'vera4castHelpers' is installed...")
    utils = importr("utils")
    if not isinstalled("remotes"):
        utils.install_packages("remotes")
    if not isinstalled("vera4castHelpers"):
        ro.r("remotes::install_github('LTREB-reservoirs/vera4castHelpers', force = TRUE)")
    importr("vera4castHelpers")
    print("R helper package is installed and ready.")

def run_workflow():
    # Step 1: Dependency checks
    check_python_dependencies()
    check_R_installation()
    ensure_r_helpers()
    
    # Step 2: Run the forecasting code from main.py
    print("Running the core forecasting code (main.py)...")
    main.main()  # This will generate and save the forecast.
    
    # Step 3: Locate the saved forecast file and validate using R helper functions.
    outdir = "model_output"
    forecast_file = None
    for fname in os.listdir(outdir):
        if fname.endswith("-LSTM.csv"):
            forecast_file = os.path.join(outdir, fname)
            break
    if forecast_file is None:
        print("No forecast file found in", outdir)
        sys.exit(1)
    print("Found forecast file:", forecast_file)
    
    print("Validating forecast output using R helper...")
    ro.r(f"vera4castHelpers::forecast_output_validator(forecast_file='{forecast_file}')")
    print("Forecast validation complete.")
    # Submit the forecast by the following line:
    ro.r(f"vera4castHelpers::submit(forecast_file='{forecast_file}', first_submission=FALSE)")
    print("Forecast submission complete.")

if __name__ == "__main__":
    run_workflow()

