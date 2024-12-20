# Sentiment Analysis Web Application

## Project Overview
A machine learning-powered web application for sentiment analysis using Recurrent Neural Networks (RNN) with Flask backend and modern JavaScript frontend.

## Prerequisites
- Python 3.8+
- pip
- (Optional) Virtual Environment

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://your-repository-url.git
cd sentiment-analysis-project
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download `training.1600000.processed.noemoticon.csv`
- Place in the project root directory

### 5. Run the Application
```bash
python sentiment_analysis_app.py
```

### 6. Access the Application
- Open `index.html` in your web browser
- Ensure Flask backend is running

## Project Structure
```
sentiment-analysis-project/
│
├── sentiment_analysis_app.py   # Flask Backend
├── index.html                  # Frontend HTML
├── app.js                      # Frontend JavaScript
├── styles.css                  # Frontend Styling
│
├── requirements.txt            # Python Dependencies
├── setup.py                    # Package Setup
│
└── data/
    └── training.1600000.processed.noemoticon.csv
```

## Development

### Training the Model
- The model is trained automatically when the Flask app starts
- Modify hyperparameters in `sentiment_analysis_app.py`

### Frontend Customization
- Edit `index.html`, `app.js`, and `styles.css` for UI changes

## Deployment Considerations
- Use gunicorn or waitress for production deployment
- Set `debug=False` in Flask app
- Configure environment variables for sensitive settings

## Performance Optimization
- GPU acceleration supported via PyTorch
- Adjust batch size and model parameters as needed
