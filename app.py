from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

model = None
scaler = None

def load_model():
    global model, scaler
    try:
        with open('sleep_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

load_model()

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Sleep Quality Prediction API',
        'model_loaded': model is not None
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        features = {
            'Age': float(data.get('age', 0)),
            'Sleep duration': float(data.get('sleep_duration', 0)),
            'REM sleep percentage': float(data.get('rem_sleep_percentage', 0)),
            'Deep sleep percentage': float(data.get('deep_sleep_percentage', 0)),
            'Light sleep percentage': float(data.get('light_sleep_percentage', 0)),
            'Awakenings': int(data.get('awakenings', 0)),
            'Caffeine consumption': float(data.get('caffeine_consumption', 0)),
            'Alcohol consumption': float(data.get('alcohol_consumption', 0)),
            'Exercise frequency': float(data.get('exercise_frequency', 0)),
            'Gender_Male': 1 if data.get('gender', '').lower() == 'male' else 0,
            'Smoking status_Yes': 1 if data.get('smoking_status', '').lower() == 'yes' else 0
        }
        
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        prediction = max(0.0, min(1.0, prediction))
        
        quality = "Excellent" if prediction >= 0.85 else "Good" if prediction >= 0.75 else "Fair" if prediction >= 0.65 else "Poor"
        
        recommendations = []
        if features['Sleep duration'] < 7:
            recommendations.append({'category': 'Sleep Duration', 'recommendation': 'Aim for 7-9 hours', 'impact': 'high'})
        if features['Caffeine consumption'] > 200:
            recommendations.append({'category': 'Caffeine', 'recommendation': 'Reduce caffeine after 2 PM', 'impact': 'medium'})
        if features['Alcohol consumption'] > 2:
            recommendations.append({'category': 'Alcohol', 'recommendation': 'Limit to 1-2 drinks', 'impact': 'medium'})
        if features['Smoking status_Yes'] == 1:
            recommendations.append({'category': 'Smoking', 'recommendation': 'Consider cessation programs', 'impact': 'high'})
        if len(recommendations) == 0:
            recommendations.append({'category': 'Maintenance', 'recommendation': 'Keep up the good habits!', 'impact': 'none'})
        
        return jsonify({
            'success': True,
            'prediction': {
                'sleep_efficiency': round(prediction, 3),
                'sleep_efficiency_percentage': round(prediction * 100, 1),
                'quality_rating': quality,
                'confidence': 0.95
            },
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
