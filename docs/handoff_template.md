# Model Handoff Template: DS → Dev

## Model Information
- **Model Name**: Hybrid Fraud Detection Model
- **Version**: 1.0.0
- **Date**: [Date]
- **Model Type**: Ensemble (Logistic Regression + Random Forest)

## Model Performance
- **Accuracy**: [Value]
- **Precision**: [Value]
- **Recall**: [Value]
- **F1 Score**: [Value]
- **ROC-AUC**: [Value]

## Model Location
- **Path**: `models/hybrid/hybrid_model.pkl`
- **Registry**: `models/model_registry.json`

## Input Requirements
- **Feature Count**: [Number]
- **Feature Names**: [List]
- **Data Types**: [Specify]
- **Preprocessing Required**: [Yes/No]

## API Endpoints
- **Health Check**: `GET /health`
- **Prediction**: `POST /predict`
- **Request Format**: See `deployment/api/app.py`

## Dependencies
- Python 3.9+
- See `requirements.txt` and `deployment/api/requirements.txt`

## Deployment Instructions
1. Build Docker image: `docker build -f deployment/docker/Dockerfile .`
2. Run container: `docker-compose -f deployment/docker/compose.yml up`
3. Test endpoint: `curl http://localhost:8000/health`

## Monitoring
- Model performance metrics
- API response times
- Error rates

## Contact
- **Data Science Lead**: [Name/Email]
- **DevOps Lead**: [Name/Email]

