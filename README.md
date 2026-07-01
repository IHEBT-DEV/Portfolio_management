# Quantitative Portfolio Management & Analytics Engine

A production-grade, containerized quantitative finance and data engineering platform built in Python. This platform implements Advanced Portfolio Theory, Single-Index Risk Decomposition, and vectorized technical momentum overlays over a high-concurrency polyglot data layer (MongoDB and Redis). Fully automated with an isolated multi-stage container environment and continuous integration pipelines.

## Project Origin & Status

This repository is based on a legacy quantitative project codebase that has been systematically re-engineered from the ground up to eliminate technical debt, enforce strict type safety, and stabilize computational data models. The application architecture is currently under active production construction as new mathematical variance modules and risk metrics are continuously integrated.

## System Architecture & Data Flow

The platform implements a highly decoupled Model-View-Controller (MVC) layout separating business execution logic from HTTP interface routing gates:

1. **Ingestion Layer (services/data_extraction):** Dual-mode extraction engine utilizing requests for real-time synchronous tracking queries and aiohttp coupled with bounded asyncio.Semaphore throttling tokens to stream historical data points concurrently without rate-limit disruptions.
2. **Transformation Layer (services/data_transformation):** Standardizes raw JSON payloads into uniform time-series matrices. Enforces type-safe boundaries via IntervalEnum attributes and processes timezone-aware UTC datetime tracking maps.
3. **Storage Layer (services/data_loading):** Polyglot database persistence strategy. Employs pymongo with asynchronous-safe execution barriers for robust document indexing, and a persistent redis connection socket pool with automated string-response decoding for sub-1ms state lookups.
4. **Analytics Layer (services/data_analysis):** The quantitative core. Leverages NumPy and SciPy vectorization to evaluate optimal asset weightings along the Markowitz Efficient Frontier (SLSQP minimization models), calculate Portfolio Beta, and compute momentum indicators (SMA, EMA, Bollinger Bands, ATR, Keltner Channels).

## Project Directory Tree

```text
Portfolio_management/
├── .github/workflows/
│   └── ci.yml               # Automated CI/CD GitHub Actions Pipeline
├── modules/
│   ├── PortfolioManagementController.py  # Primary Web Application Factory
│   ├── SpotController.py                 # REST API Routing Gateways
│   └── TradingStrategiesController.py    # Automated Strategy Signal Endpoints
├── services/
│   ├── data_analysis/       # Quantitative Math & Modeling Services
│   ├── data_extraction/     # In-Memory Synchronous & Concurrent Async Extractors
│   ├── data_loading/        # Polyglot Storage Connectors (MongoDB/Redis Cache)
│   └── data_transformation/ # Data Schema Parsers and Interval Enums
├── test/
│   ├── conftest.py          # Centralized Shared Testing Mock Fixtures
│   ├── test_data_loading.py # Database & Cache State Mutation Testing Unit
│   ├── test_trading_metrics.py  # Vectorized Math & Technical Indicator Unit
│   └── test_trading_signals.py  # Algorithmic Trading Signal System Unit
├── Dockerfile               # High-Performance Multi-Stage Production Build
├── requirements.txt         # Optimized Dependency Versions Index
└── .env.example             # Zero-Hardcode Environment Blueprint Template
```

## Continuous Integration & Pipeline Automation

The repository features an automated GitHub Actions CI/CD Pipeline (ci.yml) executed on a matrix architecture. On every push or pull request to the develop or main branches, the pipeline automatically:
* Installs compiled dependencies and updates package environments.
* Runs strict static text linting checks via flake8 to enforce style compliance.
* Discovers and runs the comprehensive pytest matrix entirely in memory using structural mock isolation handlers (zero external database connection dependency required).
* Automatically runs a test-compilation verification pass over the multi-stage Dockerfile environment.

## Production Deployment (Docker Containerization)

To guarantee absolute isolation across staging and production cloud infrastructure deployments, the application utilizes a highly optimized multi-stage build process. This design splits compilation tools out from runtime layers, scaling down memory consumption while boosting infrastructure speed parameters.

To build and launch the production container cluster locally right now:

```bash
# 1. Clone the environment parameters layout
cp .env.example .env

# 2. Build the optimized multi-stage production docker image
docker build -t portfolio_management_core:latest .

# 3. Launch the container bounded securely on port 5000
docker run -d -p 5000:5000 --env-file .env portfolio_management_core:latest
```
