# Heart Disease Risk Assessment - Full Stack Application

This is a full-stack application that predicts heart disease risk using machine learning with a Flask backend API and HTML/CSS/JavaScript frontend.

## Project Structure

```
heart-risk-fullstack/
├── backend/
│   ├── app.py              # Flask API server
│   ├── model.pkl           # Trained ML model
│   ├── scaler.pkl          # Feature scaler
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Main HTML page
│   ├── style.css           # CSS styling
│   └── script.js           # JavaScript for API calls
└── README.md
```

## Setup and Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python app.py
   ```

The backend will run on `http://127.0.0.1:5000`

### Frontend Setup

1. Open `frontend/index.html` in your web browser
2. Or serve the frontend using a local server:
   ```bash
   cd frontend
   python -m http.server 8000
   ```
   Then visit `http://localhost:8000`

## Usage

1. Start the backend server (see Backend Setup above)
2. Open the frontend in your browser
3. Fill in the patient data form with the required medical parameters
4. Click "Predict Risk" to get the heart disease risk assessment
5. The result will show either "High Risk" or "Low Risk"

## API Endpoints

- `POST /predict` - Accepts patient data and returns risk prediction
  - Input: JSON with age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
  - Output: JSON with prediction ("High Risk" or "Low Risk")

## Deployment Options

### Backend Deployment
- **Render**: Upload backend/ as a Python Web Service
- **Railway**: Deploy the Flask app directly
- **Heroku**: Use the backend/ directory as the main app

### Frontend Deployment
- **Vercel**: Upload frontend/ folder
- **Netlify**: Drag and drop frontend/ folder
- **GitHub Pages**: Push frontend/ to a GitHub repository

## Model Information

The application uses a pre-trained machine learning model that was trained on heart disease data. The model takes 13 medical parameters as input and predicts the risk of heart disease.

## Features

- ✅ RESTful API with Flask
- ✅ CORS enabled for frontend communication
- ✅ Pre-trained machine learning model
- ✅ Clean, responsive frontend interface
- ✅ Real-time predictions
- ✅ Easy deployment options