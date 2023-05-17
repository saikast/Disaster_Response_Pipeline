# Disaster Message Classification Project

### Table of Contents

1. Introduction
2. File Structure
3. Setup and Installation
    1. Prerequisites
    2. Instructions
4. License
5. Screenshots

<a name="introduction"></a>

### Introduction
This project aims to build a machine learning model that can categorize disaster messages in real-time, enabling efficient routing of communications to the appropriate disaster response agencies. The project includes a web application that allows disaster response workers to input messages and receive categorization results for better decision-making.

### File Structure

1. data folder:

 * disaster_messages.csv: Dataset containing actual disaster messages (provided by Figure Eight)
 * disaster_categories.csv: Dataset containing categories for the messages
 * process_data.py: ETL pipeline script for loading, cleaning, extracting features, and storing data in SQLite database
 * DisasterResponse.db: SQLite database containing the cleaned and processed data


2. models folder:

 * train_classifier.py: Python script implementing the ML pipeline for loading cleaned data, training the model, and saving the trained model as a pickle (.pkl) file for later use
 * classifier.pkl: Pickle file containing the trained model


3. app folder:

 * run.py: Python script to launch the web app

4. screenshots folder:

 * Contains example screenshots of the web application



### Setup and Installation

#### Prerequisites
Python 3.7 or above
Libraries: Pandas, NumPy, Scikit-Learn, NLTK, SQLAlchemy, Flask, Plotly
Dataset: disaster_messages.csv, disaster_categories.csv

#### Instructions

Run the following commands in the project's root directory to set up your database and model:
1. To run ETL pipeline: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2. To run ML pipeline: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3 . Change to the app directory: cd app
4. Run the application: python run.py
5. Open a web browser and navigate to localhost:3001

### License

This project is licensed under the terms of the MIT License. See LICENSE file for more details.

https://opensource.org/license/mit/

### Screenshots

Screenshots of the web application's user interface and its functionalities will be provided in this section.
