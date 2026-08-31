# ML Pipeline

Machine learning pipeline for demand forecasting and inventory optimization.

## Components

- **ingestion/**: Data ingestion from various sources
- **preprocessing/**: Data validation and cleaning
- **features/**: Feature engineering and selection
- **training/**: Model training and optimization
- **evaluation/**: Model evaluation and validation
- **monitoring/**: Drift detection and model monitoring

## Workflow

Data → Ingestion → Preprocessing → Features → Training → Evaluation → Monitoring → Deployment

---

## Environment Setup

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Virtual environment support (built-in with Python 3.11+)

### Step 1: Create Virtual Environment

Navigate to the project root and create a virtual environment:

```bash
# On Windows
python -m venv venv

# On macOS/Linux
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt once activated.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### Step 5: Verify Installation

Run the environment verification script:

```bash
python verify_environment.py
```

This script will import all required packages and report their versions. If all imports succeed, your environment is ready for ML development.

## Included Packages

| Package | Version | Purpose |
|---------|---------|---------|
| **pandas** | 2.2.0 | Data manipulation and analysis |
| **numpy** | 1.26.4 | Numerical computing and arrays |
| **scikit-learn** | 1.4.2 | Machine learning algorithms |
| **xgboost** | 2.0.3 | Gradient boosting for regression/classification |
| **matplotlib** | 3.8.4 | Data visualization |
| **seaborn** | 0.13.2 | Statistical data visualization |
| **jupyter** | 1.0.0 | Interactive notebooks for experimentation |
| **pytest** | 7.4.4 | Unit testing framework |
| **python-dotenv** | 1.0.1 | Environment variable management |

## Future Dependencies (Coming Soon)

The following packages will be added in later phases:
- **mlflow** - Experiment tracking and model versioning
- **tensorflow** or **torch** - Deep learning frameworks
- **fastapi** - Backend API (backend component)
- **sqlalchemy** - Database ORM (backend component)

## Development Workflow

### Running Jupyter Notebooks

```bash
jupyter notebook
```

Notebooks are useful for exploration and experimentation.

### Running Tests

```bash
pytest tests/ml/
```

### Installing Additional Packages

If you need to add a new package:

1. Install it: `pip install package-name`
2. Freeze requirements: `pip freeze > requirements.txt`
3. Commit changes to version control

## Deactivating Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

**Issue**: `python: command not found` or `ModuleNotFoundError`
- **Solution**: Ensure Python 3.11+ is installed and virtual environment is activated

**Issue**: `pip: command not found`
- **Solution**: Try `python -m pip` instead of `pip`

**Issue**: Permission denied on Linux/macOS
- **Solution**: Use `pip install --user` or create virtual environment in project directory
