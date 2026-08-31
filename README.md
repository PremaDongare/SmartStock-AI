# SmartStock AI

**An Autonomous Demand Forecasting and Inventory Management Platform for E-Commerce**

> A final-year project combining machine learning, agentic AI, and modern DevOps to solve real-world inventory challenges in e-commerce.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Proposed Solution](#proposed-solution)
3. [Main Objectives](#main-objectives)
4. [System Architecture Overview](#system-architecture-overview)
5. [Major Modules](#major-modules)
6. [Technology Stack](#technology-stack)
7. [Development Phases](#development-phases)
8. [Agentic AI Component](#agentic-ai-component)
9. [MLflow & MLOps Integration](#mlflow--mlops-integration)
10. [Demand Forecasting to Inventory Pipeline](#demand-forecasting-to-inventory-pipeline)
11. [Project Structure](#project-structure)
12. [Getting Started](#getting-started)

---

## Problem Statement

E-commerce businesses face critical challenges in inventory management:

1. **Demand Uncertainty**: Manual forecasting fails to capture complex seasonal, trend, and promotional patterns
2. **Stock-Outs**: Insufficient inventory during peak demand periods leads to lost sales and customer dissatisfaction
3. **Overstock Waste**: Excess inventory ties up capital and risks obsolescence, especially for perishable or seasonal goods
4. **Manual Processes**: Current systems rely on rule-based heuristics that don't adapt to market changes
5. **Model Drift**: ML models degrade over time as market conditions, customer behavior, and product mix evolve
6. **Lack of Automation**: Model retraining and error diagnosis require manual intervention, creating operational bottlenecks

These challenges compound in multi-SKU environments where hundreds or thousands of products require individualized forecasting strategies.

---

## Proposed Solution

**SmartStock AI** is an autonomous platform that:

1. **Automates Demand Forecasting**: Uses machine learning to predict demand with high accuracy across products and time horizons
2. **Generates Inventory Recommendations**: Translates forecasts into actionable replenishment strategies (when to order, how much)
3. **Self-Heals Through Monitoring**: Detects model drift in real-time and automatically triggers retraining
4. **Diagnoses Failures Intelligently**: Employs agentic AI to analyze forecast errors, identify root causes, and propose experiment strategies
5. **Tracks Experiments Systematically**: Integrates MLflow for reproducible model development and experiment management
6. **Provides Real-Time Insights**: Delivers forecasts and inventory recommendations via a React dashboard

The platform operates **autonomously** with human oversight—models retrain automatically, drift is detected continuously, and experiments are logged systematically.

---

## Main Objectives

### Functional Objectives
- **OF1**: Implement end-to-end ML pipeline (ingestion → preprocessing → training → evaluation → monitoring)
- **OF2**: Build accurate demand forecasting models for multiple products
- **OF3**: Translate demand forecasts into inventory replenishment recommendations
- **OF4**: Provide real-time dashboard for monitoring forecasts and inventory status
- **OF5**: Detect and respond to model drift autonomously

### Technical Objectives
- **OT1**: Design scalable microservices architecture (backend + ML pipeline + frontend)
- **OT2**: Integrate MLflow for comprehensive experiment tracking and model versioning
- **OT3**: Implement agentic AI for automated error diagnosis and experiment planning
- **OT4**: Containerize services for reproducible deployment
- **OT5**: Establish CI/CD pipeline with automated testing

### Educational Objectives
- **OE1**: Demonstrate proficiency in modern ML engineering practices
- **OE2**: Showcase integration of classical ML/DL with agentic AI
- **OE3**: Design production-grade architecture suitable for enterprise deployment

---

## System Architecture Overview

SmartStock AI follows a **modular monorepo architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                              │
│                    (React Frontend)                             │
│         - Demand forecast visualization                         │
│         - Inventory dashboard                                   │
│         - Alert and anomaly display                             │
└────────────────────┬──────────────────────────────────────────┘
                     │ REST API (FastAPI)
┌────────────────────▼──────────────────────────────────────────┐
│                   BACKEND SERVICE                              │
│                   (FastAPI, PostgreSQL)                        │
│  - User authentication & authorization                         │
│  - API endpoints for forecasts & inventory                     │
│  - Database management (products, orders, forecasts)           │
│  - Orchestration of ML pipeline execution                      │
└────────────────────┬──────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌─────────┐ ┌──────────┐ ┌──────────┐
    │  Data   │ │    ML    │ │ Agentic  │
    │Pipeline │ │ Pipeline │ │   AI     │
    │         │ │          │ │          │
    │ Ingestion   Monitoring  Error
    │ Validation  MLflow      Diagnosis
    │ Features    Retraining  Experiment
    │ Processing             Planning
    └─────────┘ └──────────┘ └──────────┘
        │            │            │
        └────────────┼────────────┘
                     │
            ┌────────▼────────┐
            │   PostgreSQL    │
            │   Database      │
            └─────────────────┘
```

**Key Design Principles:**
- **Separation of Concerns**: ML, backend, and frontend are independently deployable
- **Autonomy**: Drift detection and retraining happen automatically without human intervention
- **Observability**: MLflow tracks all experiments; logs capture all decisions
- **Scalability**: Microservices can scale independently based on demand

---

## Major Modules

### 1. **Data Pipeline** (`ml/ingestion/`, `ml/preprocessing/`)
**Status**: 🔵 Planned

- **Ingestion**: Connect to e-commerce data sources (sales, inventory, returns, customer data)
- **Validation**: Check data quality, handle missing values, detect anomalies
- **Preprocessing**: Transform raw data into formats suitable for ML
- **Output**: Clean, validated datasets ready for feature engineering

### 2. **Feature Engineering** (`ml/features/`)
**Status**: 🔵 Planned

- Extract temporal features (day of week, seasonality, trends)
- Create domain-specific features (product category, promotion status, competitor pricing)
- Handle categorical variables and scaling
- Manage feature versioning with MLflow

### 3. **Model Training** (`ml/training/`)
**Status**: 🔵 Planned

- Experiment with multiple forecasting algorithms:
  - Classical: ARIMA, Exponential Smoothing
  - ML: XGBoost, LightGBM, Random Forest
  - Deep Learning: LSTM, Transformer-based models
- Hyperparameter tuning and cross-validation
- Track experiments with MLflow
- Save trained models with metadata

### 4. **Model Evaluation** (`ml/evaluation/`)
**Status**: 🔵 Planned

- Compute evaluation metrics (RMSE, MAE, MAPE)
- Backtesting on historical data
- Compare model performance against baseline
- Generate evaluation reports

### 5. **Monitoring & Drift Detection** (`ml/monitoring/`)
**Status**: 🔵 Planned

- Monitor prediction drift (predictions deviate from actual values)
- Detect data drift (input features change distribution)
- Monitor model performance degradation
- Trigger automated retraining when drift exceeds thresholds

### 6. **Demand Forecasting** (Core ML Output)
**Status**: 🔵 Planned

- Generate demand forecasts for each product across multiple time horizons (daily, weekly, monthly)
- Produce confidence intervals and uncertainty estimates
- Output format: `{product_id, timestamp, forecast_value, confidence_interval}`

### 7. **Inventory Recommendation Engine**
**Status**: 🔵 Planned

**Pipeline**: Forecast → Recommendation
```
Demand Forecast
     ↓
Calculate Safety Stock (buffer for uncertainty)
     ↓
Determine Reorder Point (when to trigger purchase order)
     ↓
Calculate Economic Order Quantity (how much to order)
     ↓
Generate Replenishment Recommendation
     ↓
Output: {product_id, reorder_point, order_quantity, urgency}
```

**Inventory Algorithms**:
- Classic EOQ model
- Service-level optimization (maintain 95% or 99% in-stock probability)
- Lead-time considerations
- Multi-echelon inventory optimization (if applicable)

### 8. **Backend API** (`backend/`)
**Status**: 🔵 Planned

- User authentication & authorization
- REST endpoints:
  - `/api/forecasts/{product_id}` - Get demand forecasts
  - `/api/inventory/recommendations` - Get replenishment recommendations
  - `/api/products` - Product catalog management
  - `/api/models` - Model version management
  - `/api/health` - System health status
- Database models (Products, Orders, Forecasts, Inventory)
- Integration with ML pipeline

### 9. **Frontend Dashboard** (`frontend/`)
**Status**: 🔵 Planned

- **Demand Forecast View**: Charts showing predicted vs actual demand
- **Inventory Dashboard**: Current stock levels, reorder points, urgency flags
- **Alerts & Anomalies**: Highlight unusual patterns or missed forecasts
- **Model Performance**: Track RMSE, MAE over time
- **Experiment Tracker**: MLflow experiment results visualization

### 10. **Agentic AI Component** (Autonomous Error Diagnosis & Experiment Planning)
**Status**: 🔵 Planned

See [Agentic AI Component](#agentic-ai-component) section below.

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, TypeScript, Plotly/Chart.js | Interactive dashboard |
| **Backend** | FastAPI (Python), Uvicorn | High-performance REST API |
| **Database** | PostgreSQL 15 | Transactional data storage |
| **ML/Data** | Python 3.11+, Pandas, NumPy, Scikit-learn | Data processing & ML |
| **Deep Learning** | TensorFlow / PyTorch | Neural network models |
| **Experiment Tracking** | MLflow | Model versioning & experiment management |
| **Containerization** | Docker, Docker Compose | Reproducible environments |
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Agentic AI** | LLM API (OpenAI / Claude / Local LLM) | Error diagnosis & planning |

---

## Development Phases

### **Phase 1: Backend Foundation** (Weeks 1-2)
- [ ] Set up FastAPI project structure
- [ ] Design database schema (Products, Orders, Inventory)
- [ ] Create basic REST API endpoints
- [ ] Set up PostgreSQL connection
- [ ] Write unit tests for backend

### **Phase 2: Data Pipeline & Feature Engineering** (Weeks 3-4)
- [ ] Implement data ingestion module
- [ ] Create data validation logic
- [ ] Build preprocessing pipeline
- [ ] Develop feature engineering functions
- [ ] Create sample datasets for testing

### **Phase 3: ML Model Development** (Weeks 5-6)
- [ ] Implement demand forecasting models
- [ ] Set up MLflow experiment tracking
- [ ] Conduct model training and evaluation
- [ ] Compare model performance
- [ ] Serialize and save best models

### **Phase 4: Inventory Recommendation Engine** (Weeks 7)
- [ ] Design inventory recommendation algorithms
- [ ] Implement safety stock calculation
- [ ] Build reorder point logic
- [ ] Create recommendation API endpoints
- [ ] Write integration tests

### **Phase 5: Monitoring & Drift Detection** (Week 8)
- [ ] Implement drift detection algorithms
- [ ] Build monitoring dashboards
- [ ] Create retraining triggers
- [ ] Set up automated retraining pipeline

### **Phase 6: Agentic AI for Error Diagnosis** (Weeks 9-10)
- [ ] Integrate LLM API
- [ ] Build error analysis agent
- [ ] Implement experiment planning agent
- [ ] Create reporting and logging

### **Phase 7: Frontend Dashboard** (Weeks 11-12)
- [ ] Design UI components
- [ ] Build demand forecast visualization
- [ ] Implement inventory dashboard
- [ ] Integrate with backend API
- [ ] Add real-time alerts

### **Phase 8: DevOps & Deployment** (Weeks 13-14)
- [ ] Dockerize backend and frontend
- [ ] Set up GitHub Actions CI/CD
- [ ] Write deployment documentation
- [ ] Integration testing
- [ ] Final optimization and polish

---

## Agentic AI Component

**Status**: 🔵 Planned

The Agentic AI component provides **autonomous error diagnosis and experiment planning**.

### How It Works

```
┌──────────────────────┐
│  Monitoring System   │
│  (Detects Issues)    │
└──────────┬───────────┘
           │
           ▼ (Triggers on high error)
┌──────────────────────┐
│  Error Analysis      │
│  Agent               │
└──────────┬───────────┘
           │ Analyzes:
           ├─ Historical errors
           ├─ Data patterns
           ├─ Model performance
           └─ Environmental changes
           │
           ▼
┌──────────────────────┐
│  Root Cause          │
│  Hypothesis          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Experiment          │
│  Planning Agent      │
└──────────┬───────────┘
           │ Proposes:
           ├─ Feature engineering ideas
           ├─ Model architecture changes
           ├─ Hyperparameter adjustments
           └─ Data collection strategies
           │
           ▼
┌──────────────────────┐
│  Experiment Log      │
│  (MLflow)            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Automated           │
│  Retraining          │
└──────────────────────┘
```

### Key Features

1. **Error Diagnosis Agent**
   - Analyzes forecast errors and categorizes them
   - Identifies temporal patterns in failures
   - Correlates errors with data or environment changes
   - Generates hypotheses about root causes

2. **Experiment Planning Agent**
   - Suggests feature engineering improvements
   - Recommends model or hyperparameter changes
   - Designs A/B tests for competing approaches
   - Logs all suggestions in MLflow for reproducibility

3. **Autonomous Workflows**
   - No human intervention required to suggest next experiments
   - All proposals are logged for audit and learning
   - Integrates with MLflow for systematic experimentation

---

## MLflow & MLOps Integration

**Status**: 🔵 Planned

MLflow provides systematic management of the ML lifecycle.

### MLflow Components in SmartStock AI

```
┌─────────────────────────────────────────────────────┐
│          MLflow Tracking Server                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Experiment 1: Initial Baseline Model              │
│  ├─ Model: Linear Regression                       │
│  ├─ Metrics: RMSE=145.3, MAE=98.2                 │
│  ├─ Params: {features: 10, lookback: 30}          │
│  ├─ Artifacts: model.pkl, training_data.csv       │
│  └─ Date: 2026-08-31                              │
│                                                     │
│  Experiment 2: Feature Engineering v1              │
│  ├─ Model: XGBoost                                 │
│  ├─ Metrics: RMSE=89.5, MAE=62.1                  │
│  ├─ Params: {n_estimators: 100, max_depth: 6}     │
│  ├─ Artifacts: model.pkl, features.pkl            │
│  ├─ Tags: {status: "candidate"}                   │
│  └─ Date: 2026-09-05                              │
│                                                     │
│  Experiment 3: LSTM Baseline                       │
│  ├─ Model: LSTM Neural Network                     │
│  ├─ Metrics: RMSE=92.1, MAE=65.3                  │
│  ├─ Params: {layers: 2, units: 64, epochs: 50}    │
│  ├─ Artifacts: model.h5, training_history.json    │
│  └─ Date: 2026-09-08                              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### MLOps Workflow

1. **Experiment Tracking**
   - Every training run logs: parameters, metrics, artifacts, and metadata
   - Enable comparison of different modeling approaches
   - Version control for models (which model is production?)

2. **Model Registry**
   - Track "Production" vs "Staging" vs "Experimental" models
   - Automatic versioning for reproducibility
   - Rollback capability if model performance degrades

3. **Automated Retraining Pipeline**
   ```
   Weekly Retraining Trigger
   ↓
   Load latest data
   ↓
   Run training script
   ↓
   Evaluate new model
   ↓
   If performance > production model:
      Register as new "Production" version
   Else:
      Log as "Experimental" for review
   ↓
   Update dashboard with new metrics
   ```

4. **Experiment Planning Integration**
   - Agentic AI proposes experiments
   - Each proposed experiment is logged with rationale
   - Results feed back into agent's learning

---

## Demand Forecasting to Inventory Pipeline

**Status**: 🔵 Planned

This section explains how demand forecasts translate into actionable inventory recommendations.

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────┐
│ DEMAND FORECASTING                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Input: Historical sales, seasonality, trends, promotions   │
│ Process: ML model predicts demand                           │
│ Output: Demand forecast with uncertainty                   │
│                                                             │
│ Example: Product SKU-12345                                 │
│  - Next 7 days: 100 ± 20 units                            │
│  - Next 30 days: 450 ± 60 units                           │
│  - Confidence: 95%                                         │
│                                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ INVENTORY OPTIMIZATION                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Input: Demand forecast + current inventory + lead time    │
│ Process: Calculate optimal reorder parameters              │
│ Output: Replenishment recommendation                       │
│                                                             │
│ Step 1: Safety Stock Calculation                           │
│   Safety Stock = Z-score × σ (demand) × √(lead time)       │
│   - Z-score = 1.65 for 95% service level                  │
│   - σ = standard deviation from forecast                   │
│   - Example: Safety Stock = 33 units                       │
│                                                             │
│ Step 2: Reorder Point (ROP)                               │
│   ROP = (Avg Demand × Lead Time) + Safety Stock            │
│   - Avg Demand = 100 units/day                            │
│   - Lead Time = 7 days                                    │
│   - ROP = (100 × 7) + 33 = 733 units                      │
│   → TRIGGER ORDER when stock falls below 733              │
│                                                             │
│ Step 3: Economic Order Quantity (EOQ)                      │
│   EOQ = √(2 × D × S / H)                                   │
│   - D = Annual demand = 36,500 units                       │
│   - S = Ordering cost = $50 per order                      │
│   - H = Holding cost = $2 per unit per year               │
│   - EOQ = √(2 × 36500 × 50 / 2) ≈ 1,518 units            │
│   → ORDER 1,518 units when ROP triggered                  │
│                                                             │
│ Step 4: Urgency Assessment                                │
│   Current Stock = 600 units                               │
│   Days to Stockout = 600 / 100 = 6 days                   │
│   Lead Time = 7 days                                      │
│   → URGENCY: CRITICAL (order immediately)                 │
│                                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ ACTIONABLE RECOMMENDATION                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Product: SKU-12345                                         │
│ Current Stock: 600 units                                  │
│ Reorder Point: 733 units (THRESHOLD)                      │
│ Recommended Order Quantity: 1,518 units                   │
│ Urgency: CRITICAL ⚠️                                       │
│ Confidence: HIGH (95%)                                     │
│                                                             │
│ Rationale:                                                 │
│  ✓ Current stock below reorder point                      │
│  ✓ Forecasted demand is high                              │
│  ✓ Lead time is longer than days to stockout              │
│  → ACTION: Place purchase order immediately               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Algorithms

| Algorithm | Purpose | Status |
|-----------|---------|--------|
| **Safety Stock (Stochastic)** | Buffer for demand variability | 🔵 Planned |
| **Economic Order Quantity (EOQ)** | Optimal order size minimizing total cost | 🔵 Planned |
| **Reorder Point (ROP)** | Trigger point for purchase orders | 🔵 Planned |
| **ABC Analysis** | Classify products by importance | 🔵 Planned |
| **Multi-Echelon Optimization** | For warehouse + retail scenarios | 🔵 Planned |

### Feedback Loop

```
Recommendation Implemented
        ↓
Inventory Adjusted
        ↓
Actual Demand Realized
        ↓
Model Performance Evaluated
        ↓
If deviation detected:
   → Trigger retraining
   → Update forecast model
   → Refine recommendation algorithm
        ↓
Next Forecast (with learned patterns)
```

---

## Project Structure

```
SmartStock-AI/
│
├── backend/                      # FastAPI backend service
│   ├── app/
│   │   ├── main.py              # Application entry point
│   │   ├── models.py            # Pydantic models (schemas)
│   │   ├── database.py          # Database connection & models
│   │   └── api/
│   │       ├── forecasts.py     # Forecast endpoints
│   │       ├── inventory.py     # Inventory recommendation endpoints
│   │       └── products.py      # Product management endpoints
│   └── requirements.txt
│
├── frontend/                     # React frontend dashboard
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.tsx
│   └── package.json
│
├── ml/                          # Machine learning pipeline
│   ├── ingestion/               # Data ingestion modules (PLANNED)
│   ├── preprocessing/           # Data validation & cleaning (PLANNED)
│   ├── features/                # Feature engineering (PLANNED)
│   ├── training/                # Model training (PLANNED)
│   ├── evaluation/              # Model evaluation (PLANNED)
│   └── monitoring/              # Drift detection (PLANNED)
│
├── data/                        # Data storage
│   ├── raw/                     # Raw source data
│   ├── processed/               # Processed/cleaned data
│   └── sample/                  # Sample datasets for testing
│
├── models/                      # Trained model artifacts
│   └── .gitkeep                 # Placeholder for Git tracking
│
├── notebooks/                   # Jupyter notebooks for exploration
│   └── .gitkeep
│
├── tests/                       # Test suite
│   ├── backend/                 # Backend API tests (PLANNED)
│   ├── ml/                      # ML pipeline tests (PLANNED)
│   └── agent/                   # Agentic AI tests (PLANNED)
│
├── scripts/                     # Utility scripts
│   ├── train.py                 # Train models (PLANNED)
│   ├── evaluate.py              # Evaluate models (PLANNED)
│   └── infer.py                 # Make predictions (PLANNED)
│
├── config/                      # Configuration files
│   └── .gitkeep
│
├── docker/                      # Docker configuration
│   ├── Dockerfile.backend       # Backend container definition
│   └── Dockerfile.frontend      # Frontend container definition
│
├── docs/                        # Project documentation
│   ├── architecture.md          # Architecture deep-dive (PLANNED)
│   ├── api_reference.md         # API documentation (PLANNED)
│   └── deployment.md            # Deployment guide (PLANNED)
│
├── .github/
│   └── workflows/               # GitHub Actions CI/CD (PLANNED)
│
├── .gitignore                   # Git ignore rules
├── docker-compose.yml           # Docker Compose setup
└── README.md                    # This file
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15
- Docker & Docker Compose
- Git

### Installation (Planned)

Detailed setup instructions will be provided as components are implemented. For now:

1. Clone the repository
2. Install dependencies (coming soon)
3. Configure environment variables (coming soon)
4. Run database migrations (coming soon)
5. Start services with Docker Compose (coming soon)

### Development Setup (Planned)

- Backend: `cd backend && pip install -r requirements.txt`
- Frontend: `cd frontend && npm install`
- ML: `cd ml && pip install -r requirements.txt`
- Tests: `pytest tests/`

---

## Current Status

### ✅ Completed
- Project structure and folder hierarchy
- Technology stack selection
- Architecture design
- Documentation

### 🔵 Planned (In Progress Order)
1. Backend FastAPI setup
2. Database schema and migrations
3. Data pipeline (ingestion, preprocessing)
4. Feature engineering
5. ML model development
6. Monitoring and drift detection
7. Agentic AI component
8. Frontend dashboard
9. DevOps and CI/CD

---

## Project Team

- **Author**: [Your Name]
- **Institution**: [Your College/University]
- **Project Type**: Final-Year Project
- **Advisors**: [If applicable]

---

## License

This project is part of an academic final-year project. Use for educational and non-commercial purposes.

---

## Acknowledgments

- SmartStock AI is inspired by real-world challenges in e-commerce inventory management
- Architecture draws on MLOps best practices and production ML systems
- Integration of agentic AI reflects emerging trends in autonomous ML systems

---

**Last Updated**: 2026-08-31  
**Version**: 1.0-initial

