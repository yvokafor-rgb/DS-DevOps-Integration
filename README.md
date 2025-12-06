# Fraud Detection Hybrid Model

A machine learning project implementing a hybrid ensemble model for fraud detection, combining Logistic Regression and Random Forest classifiers.

## Project Structure

```
fraud-detection-hybrid-model/
├── data/
│   ├── raw/                # Original uploaded dataset (untouched)
│   ├── processed/          # Cleaned, preprocessed, SMOTEd data
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_hybrid_model.ipynb
├── src/
│   ├── data_prep.py        # Cleaning, preprocessing, z-score, SMOTE
│   ├── train_logreg.py     # Model 1
│   ├── train_rf.py         # Model 2
│   ├── train_hybrid.py     # Ensemble logic
│   ├── evaluate.py         # Shared evaluation utilities
│   ├── utils.py            # Helper functions
│   └── config.py           # Central config (paths, params, seeds)
├── models/
│   ├── logreg/
│   ├── random_forest/
│   ├── hybrid/
│   └── model_registry.json  # Version tracking
├── deployment/
│   ├── api/
│   │   ├── app.py           # FastAPI endpoint for inference
│   │   └── requirements.txt
│   └── docker/
│       ├── Dockerfile
│       └── compose.yml
├── ci-cd/
│   ├── test_scripts/        # Unit tests for training functions
│   └── model_validation/    # Automated checks before deployment
├── .github/
│   └── workflows/
│       ├── ci.yml           # Runs tests, linting, style checks
│       └── cd.yml            # Optional: deploy inference API
├── docs/
│   ├── project_plan.md
│   ├── communication_plan.md
│   ├── handoff_template.md  # DS → Dev handoff (model card)
│   └── sprint_backlog.md
├── tests/
│   ├── test_data_prep.py
│   ├── test_models.py
│   └── test_api.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Features

- **Data Preprocessing**: Automated cleaning, outlier removal (z-score), and SMOTE balancing
- **Multiple Models**: Logistic Regression, Random Forest, and Hybrid Ensemble
- **Model Evaluation**: Comprehensive metrics and reporting
- **API Deployment**: FastAPI-based inference endpoint
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: Automated testing and validation
- **Model Registry**: Version tracking for trained models

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- (Optional) Docker and Docker Compose

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yvokafor-rgb/DS-DevOps-Integration.git
cd DS-DevOps-Integration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset in `data/raw/`

### Usage

#### Data Preprocessing
```python
from src.data_prep import preprocess_data

X, y = preprocess_data()
```

#### Training Models
```python
# Train individual models
from src.train_logreg import train_logistic_regression
from src.train_rf import train_random_forest
from src.train_hybrid import train_hybrid_model

# Or use the notebooks in the notebooks/ directory
```

#### Running the API
```bash
# Using Python
cd deployment/api
uvicorn app:app --host 0.0.0.0 --port 8000

# Using Docker
docker-compose -f deployment/docker/compose.yml up
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## CI/CD

The project includes GitHub Actions workflows for:
- **CI**: Automated testing, linting, and code quality checks
- **CD**: Model validation and deployment (configure as needed)

## Documentation

See the `docs/` directory for:
- Project plan
- Communication plan
- Handoff template
- Sprint backlog

## License

MIT License - see LICENSE file for details

## Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.
