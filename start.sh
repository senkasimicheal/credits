#!/bin/bash

# Install Python packages
pip install -r requirements.txt
pip install pandas
pip install numpy
pip install sklearn
# Start the Flask app
python app.py