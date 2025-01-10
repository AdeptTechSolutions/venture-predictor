# Venture Predictor

An intelligent system that leverages historical venture capital data to predict startup success using machine learning and conversational AI.

## Overview

This project combines the power of machine learning and conversational AI to create a data-driven approach for evaluating startup potential. By analyzing over 300 data points from venture-backed companies since the 1990s, the system identifies patterns and makes predictions about a startup's likelihood of success.

The project consists of two main components:
- A machine learning model trained on historical startup data
- An interactive chat interface powered by AutoGen that guides users through the evaluation process

## Features

- **Historical Data Analysis**: Processes comprehensive startup data including funding rounds, acquisitions, milestones, and more
- **Smart Preprocessing**: Handles missing values and encodes categorical variables for robust model training
- **Gradient Boosting Model**: Uses HistGradientBoostingClassifier for accurate success prediction
- **Interactive Chat Interface**: Features a VC analyst agent that collects information through natural conversation
- **Automated Evaluation**: Provides probability-based success predictions with detailed insights

## Requirements

```
autogen[gemini,ipython]
openai
pandas
numpy
scikit-learn
python-dotenv
flaml[automl]
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/venture-success-predictor.git
cd venture-success-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Training the Model

1. Ensure your data files are in the `data/` directory with the following structure:
   - objects.csv
   - acquisitions.csv
   - funding_rounds.csv
   - ipos.csv
   - investments.csv
   - milestones.csv

2. Run the training script:
```bash
python train.py
```

This will create trained models and encoders in the `models/` directory.

### Running the Chat

To start the interactive VC analysis session:
```bash
python chat.py
```

The system will initiate a conversation where the VC analyst will ask questions about your startup. Be prepared to provide information about:
- Company age
- Country of operation
- Business category
- Acquisition history
- Funding details
- Milestones and relationships
- Investment history

## How It Works

### Data Processing (`train.py`)
1. Loads and preprocesses historical startup data
2. Engineers relevant features from raw data
3. Handles missing values and encodes categorical variables
4. Trains a gradient boosting model for success prediction
5. Saves the trained model and necessary encoders

### Interactive Analysis (`chat.py`)
1. Creates a group chat with three agents:
   - User Proxy: Interfaces with the human user
   - VC Analyst: Guides the conversation and collects information
   - Executor: Handles model prediction requests
2. The VC analyst conducts a natural conversation to gather necessary data
3. Once all information is collected, the system provides a detailed prediction

## Project Structure

```
venture-success-predictor/
├── chat.py            # Interactive chat interface
├── train.py           # Model training script
├── requirements.txt   # Project dependencies
├── .env               # Environment variables (create this)
├── data/              # Dataset directory
└── models/            # Trained models directory
```