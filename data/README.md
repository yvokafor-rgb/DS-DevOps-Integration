# Data Directory

## Structure
- `raw/`: Original uploaded dataset (untouched)
- `processed/`: Cleaned, preprocessed, SMOTEd data

## Data Files
Place your raw dataset in the `raw/` directory. The preprocessing pipeline will generate processed data in the `processed/` directory.

## Data Processing Pipeline
1. Load raw data from `raw/`
2. Clean data (handle missing values, duplicates)
3. Remove outliers using z-score method
4. Apply SMOTE for class balancing
5. Save processed data to `processed/`

## Usage
```python
from src.data_prep import preprocess_data

X, y = preprocess_data()
```

