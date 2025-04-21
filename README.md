# Py_Vera

This is a Python–R pipeline for the VERA Forecasting Challenge. It Provides: 


- 📥**Data Acquisition & Preprocessing**  
  Automatically fetches online in‑situ and does necesaary preprocsessing steps to prepares it for modeling.

- 🤖**LSTM Model Training**  
  Trains a Long Short Term Memory model to forecast Chlorophyll‑a concentrations (Ensemble).

- ✅**Validation & Submission**  
  Utlizes the R package `vera4castHelpers` to validate forecasts and format submissions to the challenge.

- ⏰**Automated Daily Forecasts**  
  Schedules and runs the full workflow each day via GitHub Actions.
