#!/bin/bash
source venv/bin/activate

pkill -f streamlit

streamlit run app.py --server.headless true --server.port 8501 &
