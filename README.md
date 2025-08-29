# Credit Default Risk Prediction

## Project Overview
A comprehensive financial analysis system for predicting corporate credit default risk using real-world financial data from the Financial Modeling Prep (FMP) API.

## Features
- **Real-time Data**: Live financial data from FMP API
- **Risk Assessment**: Industry-standard credit risk analysis
- **Multiple Models**: Advanced scoring algorithms
- **Web Interface**: Professional Flask-based web application
- **Interactive Charts**: Real-time visualizations with Chart.js
- **Financial Analysis**: Based on real credit risk assessment methodologies

## Technology Stack
- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Data**: FMP API, pandas, numpy
- **Visualization**: Chart.js
- **Deployment**: Flask development server

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd credit_default_prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your FMP API key
FMP_API_KEY=your_api_key_here
```

4. **Run the application**:
```bash
python website.py
```

## Usage
1. Open your browser and go to `http://localhost:5000`
2. Use the Risk Prediction tab for manual financial data input
3. Use the Company Analysis tab for real-time company analysis
4. View detailed risk assessments and interactive charts

## Deployment

### Environment Variables
- `FMP_API_KEY`: Your Financial Modeling Prep API key
- `FLASK_ENV`: Set to 'production' for deployment
- `FLASK_DEBUG`: Set to 'False' for production

### For Netlify/Vercel Deployment
1. Set environment variables in your hosting platform
2. Ensure all API endpoints are properly configured
3. Update any hardcoded localhost URLs for production

## API Key Setup
Get your free FMP API key at: https://financialmodelingprep.com/developer/docs

## Project Structure
```
credit_default_prediction/
├── data/                 # Data storage
├── templates/           # HTML templates
├── website.py          # Main Flask application
├── config.py           # Configuration settings
├── utils.py            # Utility functions
├── data_collection.py  # Data collection module
└── requirements.txt    # Python dependencies
```

## License
This project is for educational and demonstration purposes.