# 🚀 Quick Start Guide - Credit Default Risk Prediction

## ⚡ Get Started in 5 Minutes

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Set Up API Key**
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env and add your FMP API key
# Get free API key from: https://financialmodelingprep.com/developer/docs/
```

### 3. **Run the Complete Pipeline**
```bash
# Execute the entire ML pipeline
python main.py
```

### 4. **Start the API (Optional)**
```bash
# Start Flask API after pipeline completion
python main.py --start-api

# Or start API separately
python app.py
```

### 5. **Launch the GUI (NEW!)**
```bash
# Launch the modern web interface
python launch_gui.py

# Or run Streamlit directly
streamlit run gui.py
```

## 🎯 Quick Test Commands

### **Test API Health**
```bash
curl http://localhost:5000/health
```

### **Make a Prediction**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

### **Batch Predictions**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL"]}'
```

### **Access GUI**
```bash
# Open in browser: http://localhost:8501
# The GUI will automatically open in your default browser
```

## 📊 What You'll Get

### **Data Collection Results**
- Company financial data from FMP API
- Financial statements and ratios
- Data quality reports

### **ML Model Performance**
- Trained models (Logistic Regression, Random Forest, XGBoost)
- Performance metrics (AUC-ROC, Precision, Recall)
- Hyperparameter optimization results

### **Visualizations**
- ROC curves and confusion matrices
- Feature importance plots
- Model comparison charts

### **API Service**
- Real-time prediction endpoints
- Company risk assessments
- Batch processing capabilities
- Comprehensive documentation

### **Modern Web GUI (NEW!)**
- Interactive financial data input
- Real-time risk predictions
- Beautiful visualizations with Plotly
- Company analysis with historical data
- Professional, responsive design

## 🔧 Customization Options

### **Skip Pipeline Steps**
```bash
# Skip data collection (use existing data)
python main.py --skip-collection

# Skip preprocessing
python main.py --skip-preprocessing

# Skip training
python main.py --skip-training

# Skip evaluation
python main.py --skip-evaluation
```

### **Run Individual Modules**
```bash
# Data collection only
python data_collection.py

# Data preprocessing only
python data_preprocessing.py

# Model training only
python model_training.py

# Model evaluation only
python model_evaluation.py

# GUI only
python launch_gui.py
```

## 📁 Output Locations

- **Raw Data**: `data/raw/`
- **Processed Data**: `data/processed/`
- **Trained Models**: `models/`
- **Visualizations**: `data/processed/plots/`
- **Logs**: `logs/`
- **GUI**: `http://localhost:8501`

## 🚨 Troubleshooting

### **Common Issues**

1. **API Key Error**
   - Ensure `.env` file exists with `FMP_API_KEY=your_key_here`
   - Verify API key is valid at FMP website

2. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Data Not Found**
   - Run `python main.py` to collect data first
   - Check `data/` directory exists

4. **Models Not Found**
   - Ensure model training completed successfully
   - Check `models/` directory for `.pkl` files

5. **GUI Not Loading**
   - Install Streamlit: `pip install streamlit`
   - Install Plotly: `pip install plotly`
   - Check if port 8501 is available

### **Get Help**
- Check logs in `logs/` directory
- Review error messages in console output
- Verify file permissions and directory structure

## 🎉 Success Indicators

✅ **Pipeline Complete**: All 4 steps show success  
✅ **Models Saved**: Files appear in `models/` directory  
✅ **API Running**: Health check returns "healthy" status  
✅ **Predictions Working**: API responds to prediction requests  
✅ **GUI Accessible**: Web interface loads at localhost:8501  

## 🚀 Next Steps

1. **Explore the API**: Visit `http://localhost:5000` for documentation
2. **Use the GUI**: Launch with `python launch_gui.py`
3. **Test Predictions**: Try different company tickers
4. **Review Results**: Check generated visualizations and reports
5. **Customize**: Modify `config.py` for different settings
6. **Scale Up**: Process more companies or add new features

## 🌟 **NEW: Modern Web GUI Features**

The new Streamlit GUI provides:

- **📊 Interactive Dashboard**: Clean, professional interface
- **🎯 Real-time Predictions**: Instant risk assessment
- **📈 Beautiful Visualizations**: Plotly charts and graphs
- **🏢 Company Analysis**: Historical data and trends
- **📱 Responsive Design**: Works on desktop and mobile
- **🔧 Easy Configuration**: User-friendly settings

### **GUI Usage**
1. **Launch**: `python launch_gui.py`
2. **Input Data**: Enter financial metrics manually
3. **Fetch Data**: Use company ticker for API data
4. **View Results**: Interactive charts and analysis
5. **Export**: Save results and visualizations

---

**Ready to go?** 
- Run `python main.py` for the ML pipeline
- Run `python launch_gui.py` for the modern web interface

Watch the magic happen! 🎯✨
