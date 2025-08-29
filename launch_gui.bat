@echo off
echo Starting Credit Default Risk Prediction GUI...
echo.
echo The GUI will open in your default web browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server when done.
echo.
python -m streamlit run gui.py --server.port 8501 --server.address localhost
pause
