# ⚡ Scrybe AI: Backend Engine

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-beta-blue.svg)](https://scrybe.ai)
[![GitHub Actions](https://github.com/hriday29/scrybe-ai-backend/actions/workflows/daily-analysis.yml/badge.svg)](https://github.com/hriday29/scrybe-ai-backend/actions)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**The intelligent core powering Scrybe AI's financial analysis platform. A sophisticated backend service combining cutting-edge AI with robust data processing to deliver institutional-grade market intelligence.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Technology Stack](#-technology-stack)
- [Key Features](#-key-features)
- [Getting Started](#-getting-started)
- [Project Architecture](#-project-architecture)
- [Usage](#-usage)
- [Project Roadmap](#-project-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

Scrybe AI Backend Engine represents the pinnacle of quantitative financial analysis, leveraging frontier artificial intelligence to navigate the complexities of modern stock markets. Our mission is to democratize institutional-grade trading intelligence by providing traders and investors with disciplined, data-driven insights through high-probability swing trading setups.

At the heart of our platform lies a sophisticated backend architecture that seamlessly integrates AI-powered analysis with robust data processing pipelines, delivering real-time market intelligence through our proprietary 7-Layer AI protocol.

### Core Responsibilities

- **🔄 Automated Daily Analysis:** Comprehensive daily processing of Nifty 50 stocks with intelligent market data aggregation
- **🧠 AI-Powered Intelligence:** Proprietary "Scrybe Score" generation using advanced machine learning models
- **🔌 Secure API Services:** Enterprise-grade REST API serving authenticated data to frontend applications
- **👥 User Management:** Comprehensive user data and trade logging system
- **📊 Performance Analytics:** Historical backtesting and strategy validation engine

---

## 🛠️ Technology Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)

### Core Technologies

- **Language:** Python 3.11
- **API Framework:** Flask with Gunicorn
- **AI/ML Platform:** Google Gemini Models
- **Database:** MongoDB with Atlas Cloud
- **Data Processing:** Pandas, NumPy, pandas_ta
- **Market Data:** yfinance, Alpha Vantage
- **Visualization:** Matplotlib, Plotly
- **CI/CD:** GitHub Actions
- **Authentication:** JWT with Flask-Security
- **Task Scheduling:** APScheduler

---

## ✨ Key Features

### 🤖 AI-Powered Analysis Engine
Revolutionary artificial intelligence system featuring:
- **7-Layer AI Protocol:** Multi-dimensional analysis framework
- **Scrybe Score Generation:** Proprietary confidence rating algorithm
- **DVM Ratings:** Dynamic Value Momentum indicators
- **Gemini Integration:** Advanced natural language processing for market insights
- **Adaptive Learning:** Continuous model refinement based on market performance

### 📈 Automated Market Analysis
Comprehensive daily processing system:
- **Nifty 50 Coverage:** Complete automated analysis of top Indian stocks
- **Multi-Timeframe Analysis:** Integration across multiple trading timeframes
- **Technical Indicators:** 50+ technical analysis indicators and patterns
- **Fundamental Integration:** Key financial metrics and ratios analysis
- **Market Sentiment:** News and social media sentiment integration

### 🔧 Resilient Infrastructure
Enterprise-grade system architecture:
- **APIKeyManager:** Intelligent API key rotation and rate limit management
- **High Availability:** Fault-tolerant design with automatic failover
- **Scalable Processing:** Distributed computing capabilities
- **Real-time Monitoring:** Comprehensive logging and performance metrics
- **Security First:** End-to-end encryption and secure data handling

### 🗄️ Data Management System
Robust data pipeline architecture:
- **MongoDB Integration:** High-performance document database
- **Historical Data:** Comprehensive market data storage and retrieval
- **User Management:** Secure user authentication and authorization
- **Trade Logging:** Complete transaction history and portfolio tracking
- **Backup & Recovery:** Automated data backup and disaster recovery

### 📊 Backtesting Engine
Advanced strategy validation platform:
- **Historical Simulation:** Parameterized backtesting across multiple timeframes
- **Performance Metrics:** Comprehensive risk and return analysis
- **Strategy Optimization:** AI-driven parameter tuning and optimization
- **Risk Management:** Advanced position sizing and risk control algorithms
- **Visualization Tools:** Interactive performance charts and analytics

### 🔐 Secure REST API
Professional-grade API infrastructure:
- **RESTful Design:** Industry-standard API architecture
- **JWT Authentication:** Secure token-based authentication
- **Rate Limiting:** Intelligent request throttling and management
- **API Documentation:** Comprehensive OpenAPI/Swagger documentation
- **Versioning Support:** Backward-compatible API versioning

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.11 or higher
- MongoDB (local or Atlas cloud instance)
- Git version control
- Virtual environment manager (venv/conda)

### Installation

1. **Clone the repository** (authorized personnel only)
   ```bash
   git clone https://github.com/hriday29/scrybe-ai-backend.git
   cd scrybe-ai-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Update your `.env` file with the following configuration:
   ```ini
   # Database Configuration
   ANALYSIS_DB_URI="mongodb://localhost:27017/scrybe_analysis"
   SCHEDULER_DB_URI="mongodb://localhost:27017/scrybe_scheduler"
   
   # Google Gemini API Keys
   GOOGLE_API_KEY_1="your_primary_gemini_key"
   GOOGLE_API_KEY_2="your_secondary_gemini_key"
   GOOGLE_API_KEY_3="your_tertiary_gemini_key"
   
   # Security Configuration
   SECRET_KEY="your_flask_secret_key"
   JWT_SECRET_KEY="your_jwt_secret_key"
   
   # External APIs
   ALPHA_VANTAGE_KEY="your_alpha_vantage_key"
   ```

5. **Initialize the database**
   ```bash
   python scripts/init_database.py
   ```

6. **Start the development server**
   ```bash
   python run_api.py
   ```

The API will be available at `http://localhost:5001`

### Quick Start Commands

```bash
# Run daily analysis
python run_daily_jobs.py

# Start API server
python run_api.py

# Run backtesting
python historical_runner.py --start-date 2023-01-01 --end-date 2024-01-01

# Run tests
python -m pytest tests/

# Generate API documentation
python scripts/generate_docs.py
```

---

## 🏗️ Project Architecture

Our backend follows a microservices-inspired architecture with clear separation of concerns:

```
scrybe-ai-backend/
├── api/
│   ├── __init__.py
│   ├── index.py              # Main Flask application
│   ├── auth.py              # Authentication routes
│   ├── analysis.py          # Analysis endpoints
│   └── utils.py             # API utilities
├── core/
│   ├── ai_analyzer.py       # AI analysis engine
│   ├── data_retriever.py    # Market data fetching
│   ├── database_manager.py  # Database operations
│   └── api_key_manager.py   # API key rotation
├── analysis/
│   ├── technical.py         # Technical analysis
│   ├── fundamental.py       # Fundamental analysis
│   └── sentiment.py         # Sentiment analysis
├── backtesting/
│   ├── backtester.py        # Backtesting engine
│   ├── historical_runner.py # Historical simulation
│   └── metrics.py           # Performance metrics
├── utils/
│   ├── logger.py            # Logging configuration
│   ├── validators.py        # Input validation
│   └── helpers.py           # Utility functions
└── tests/                   # Test suites
```

### Core Components

#### 🧠 AI Analysis Engine (`ai_analyzer.py`)
The heart of our AI-powered analysis system:
- **Model Integration:** Seamless interface with Google Gemini models
- **Prompt Engineering:** Optimized prompts for financial analysis
- **Response Processing:** Structured data extraction from AI responses
- **Error Handling:** Robust error recovery and fallback mechanisms

#### 📊 Data Retrieval System (`data_retriever.py`)
Comprehensive market data aggregation:
- **Multi-Source Integration:** yfinance, Alpha Vantage, and custom sources
- **Data Validation:** Automatic data quality checks and cleaning
- **Caching Layer:** Intelligent caching for improved performance
- **Rate Limit Management:** Automated handling of API limitations

#### 🗄️ Database Manager (`database_manager.py`)
Centralized data persistence layer:
- **MongoDB Operations:** Optimized database queries and operations
- **Connection Pooling:** Efficient database connection management
- **Data Modeling:** Structured schemas for all data types
- **Migration Support:** Database schema versioning and migrations

#### 🔄 Daily Jobs Orchestrator (`run_daily_jobs.py`)
Main automation engine:
- **Scheduled Analysis:** Daily market analysis automation
- **Error Recovery:** Automatic retry mechanisms for failed jobs
- **Notification System:** Alert system for job status updates
- **Performance Monitoring:** Comprehensive job execution metrics

---

## 📖 Usage

### Development Commands

```bash
# Development server with auto-reload
python run_api.py --debug

# Run specific analysis for a stock
python scripts/analyze_stock.py --symbol RELIANCE

# Export analysis data
python scripts/export_data.py --format json --date 2024-01-01

# Database management
python scripts/db_maintenance.py --cleanup --optimize

# Generate performance report
python scripts/generate_report.py --period monthly
```

### API Endpoints

#### Authentication
```bash
POST /api/auth/login
POST /api/auth/register
POST /api/auth/refresh
POST /api/auth/logout
```

#### Analysis Data
```bash
GET /api/analysis/stocks
GET /api/analysis/stock/{symbol}
GET /api/analysis/scores
GET /api/analysis/history
```

#### User Management
```bash
GET /api/user/profile
PUT /api/user/profile
GET /api/user/trades
POST /api/user/trades
```

### Environment Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `ANALYSIS_DB_URI` | MongoDB connection string for analysis data | Yes |
| `SCHEDULER_DB_URI` | MongoDB connection string for scheduling | Yes |
| `GOOGLE_API_KEY_*` | Google Gemini API keys (multiple for rotation) | Yes |
| `SECRET_KEY` | Flask application secret key | Yes |
| `JWT_SECRET_KEY` | JWT token signing key | Yes |
| `ALPHA_VANTAGE_KEY` | Alpha Vantage API key | No |
| `DEBUG` | Enable debug mode | No |

---

## 🗺️ Project Roadmap

### Q1 2025
- [ ] Enhanced AI model integration with GPT-4
- [ ] Real-time WebSocket data streaming
- [ ] Advanced risk management algorithms
- [ ] Multi-exchange support expansion

### Q2 2025
- [ ] Machine learning model interpretability
- [ ] Advanced portfolio optimization
- [ ] Options analysis integration
- [ ] Enhanced backtesting metrics

### Q3 2025
- [ ] Cryptocurrency market analysis
- [ ] Global market expansion
- [ ] Advanced alert system
- [ ] Mobile API optimizations

### Future Releases
- [ ] Quantum computing integration
- [ ] Advanced ML model ensemble
- [ ] Institutional API features
- [ ] Advanced risk analytics platform

---

## 🤝 Contributing

We appreciate your interest in Scrybe AI! However, this is a proprietary, closed-source project and we do not accept external contributions at this time.

### For Internal Team Members

If you are part of the Scrybe AI development team:

1. **Development Standards**
   - Follow PEP 8 Python style guidelines
   - Write comprehensive unit tests (pytest)
   - Use type hints for all function signatures
   - Document all modules with docstrings

2. **Code Review Process**
   - Create feature branches from `develop`
   - All changes require peer review
   - Security review for authentication changes
   - Performance review for data processing changes

3. **Testing Requirements**
   ```bash
   # Run full test suite
   python -m pytest tests/ -v --cov=core --cov-report=html
   
   # Run specific test categories
   python -m pytest tests/unit/ -v
   python -m pytest tests/integration/ -v
   python -m pytest tests/api/ -v
   ```

4. **Documentation Standards**
   - Update API documentation for endpoint changes
   - Maintain architectural decision records (ADRs)
   - Update README for significant feature additions

For questions about internal development processes, please contact the backend team lead.

---

## 📄 License

**Proprietary License**

This software and its associated documentation, algorithms, AI models, and methodologies are proprietary to Scrybe AI and are protected by international copyright law. All rights reserved.

**Usage Restrictions:**
- This software is for authorized use only by Scrybe AI personnel
- Redistribution, modification, or derivative works are strictly prohibited
- Reverse engineering, decompilation, or disassembly is not permitted
- Commercial use outside of Scrybe AI is expressly forbidden
- AI models, prompts, and analysis methodologies are trade secrets

**Financial Disclaimer:**
All analyses, data, and signals provided by Scrybe AI are for informational and educational purposes only and should not be construed as financial advice. Trading and investing in financial markets involve substantial risk of loss and is not suitable for every investor. You should not trade with money that you cannot afford to lose. Past performance is not indicative of future results.

**Technical Disclaimer:**
The software is provided "as is" without warranty of any kind. Scrybe AI disclaims all warranties, whether express or implied, including but not limited to implied warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall Scrybe AI be liable for any damages arising from the use of this software.

---

## 📞 Contact

### Development Team
- **Backend Lead:** [backend-lead@scrybe-ai.com](mailto:backend-lead@scrybe-ai.com)
- **AI/ML Engineer:** [ai-team@scrybe-ai.com](mailto:ai-team@scrybe-ai.com)
- **DevOps Engineer:** [devops@scrybe-ai.com](mailto:devops@scrybe-ai.com)

### Technical Support
- **API Support:** [api-support@scrybe-ai.com](mailto:api-support@scrybe-ai.com)
- **Database Issues:** [db-support@scrybe-ai.com](mailto:db-support@scrybe-ai.com)
- **Integration Help:** [integration@scrybe-ai.com](mailto:integration@scrybe-ai.com)

### Business Inquiries
- **General:** [hello@scrybe-ai.com](mailto:hello@scrybe-ai.com)
- **Partnerships:** [partnerships@scrybe-ai.com](mailto:partnerships@scrybe-ai.com)

### Resources
- **API Documentation:** [docs.scrybe-ai.com/api](https://docs.scrybe-ai.com/api)
- **Architecture Guide:** [docs.scrybe-ai.com/architecture](https://docs.scrybe-ai.com/architecture)
- **Status Page:** [status.scrybe-ai.com](https://status.scrybe-ai.com)

---

**© 2025 Scrybe AI. All rights reserved.**

*Engineered with 🧠 by the Scrybe AI backend team*