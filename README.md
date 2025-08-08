# Scrybe AI - Backend API

<div align="center">

**The server-side engine that powers the Scrybe AI analysis platform**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=github&logoColor=white)
![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen?style=for-the-badge&logo=codecov&logoColor=white)
![License](https://img.shields.io/badge/License-Proprietary-red.svg?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)
![Google AI](https://img.shields.io/badge/Google_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Sentry](https://img.shields.io/badge/Sentry-362D59?style=for-the-badge&logo=sentry&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

</div>

---

## 📋 Table of Contents

- [🚀 About The Project](#-about-the-project)
- [🏗️ Core Architecture](#️-core-architecture)
- [🛠️ Technology Stack](#️-technology-stack)
- [⚙️ Key System Components](#️-key-system-components)
- [🏁 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [⚖️ License](#️-license)
- [📞 Contact](#-contact)

---

## 🚀 About The Project

This repository contains the backend service for the Scrybe AI platform. This API serves as the core engine responsible for all data processing, analysis, and user management. It acts as the single source of truth for the React frontend, providing all necessary data via a secure and efficient RESTful API.

The system is engineered for **reliability** and **scalability**, featuring automated daily jobs, robust error monitoring, and a secure authentication layer to protect user-specific endpoints. Built with modern Python technologies and cloud-native architecture, it delivers high-performance financial analysis powered by cutting-edge AI.

### ✨ Key Features

- **🔒 Secure Authentication** - Firebase-based JWT token verification
- **🤖 AI-Powered Analysis** - Integration with Google's Generative AI (Gemini)
- **⏰ Automated Processing** - Daily analysis pipeline via GitHub Actions
- **📊 Real-time Monitoring** - Comprehensive error tracking with Sentry
- **☁️ Cloud-Native** - Deployed on Render with MongoDB Atlas
- **🔄 Automated Backups** - Weekly database backups for data protection

---

## 🏗️ Core Architecture

The backend is built as a modular Flask application with clear separation of concerns:

### 🌐 **API Layer (`api/index.py`)**
Defines all public and protected REST endpoints, handles incoming requests, and manages rate limiting.

### 🧠 **Service Layer (`ai_analyzer.py`, `data_retriever.py`)**
Contains the core business logic, including integration with Google's Generative AI and external financial data sources.

### 🗄️ **Data Layer (`database_manager.py`)**
Manages all interactions with the MongoDB Atlas database clusters.

### 🚀 **Automation (`.github/workflows`)**
GitHub Actions are used for scheduled tasks, including running the daily analysis pipeline and performing automated database backups.

---

## 🛠️ Technology Stack

### **Core Technologies**
- **Framework:** Flask - Lightweight and flexible web framework
- **Language:** Python 3.9+ - Modern, powerful programming language
- **Database:** MongoDB - NoSQL database for flexible data storage

### **Authentication & Security**
- **Authentication:** Firebase Admin SDK - Secure JWT verification
- **Error Monitoring:** Sentry - Real-time error tracking and performance monitoring

### **AI & Data Processing**
- **AI Engine:** Google Generative AI (Gemini) - Advanced language model for analysis
- **Data Processing:** PyMongo - MongoDB driver for Python

### **Deployment & Operations**
- **Hosting:** Render - Cloud platform with automatic scaling
- **Server:** Gunicorn - WSGI HTTP Server
- **CI/CD:** GitHub Actions - Automated workflows and cron jobs

---

## ⚙️ Key System Components

### 🔐 **Secure Authentication**
Protects user-specific endpoints by verifying Firebase ID Tokens passed in the `Authorization` header, ensuring data privacy and security.

### 📈 **Automated Daily Analysis**
A scheduled GitHub Action (`run_daily_jobs.py`) runs the complete analysis pipeline for all tracked stocks every trading day, keeping data fresh and relevant.

### 💾 **Automated Backups**
A scheduled GitHub Action (`database-backup.yml`) performs a full `mongodump` of the production databases weekly, storing artifacts securely for disaster recovery.

### 🚨 **Real-time Error Monitoring**
The Sentry SDK captures and reports any unhandled exceptions that occur on the live server, enabling rapid response to issues.

---

## 🏁 Getting Started

### **Prerequisites**

Ensure you have the following installed:
- **Python 3.9+** and pip
- Access to all necessary API keys and database URIs
- Git for version control

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/scrybe-ai-backend.git
   cd scrybe-ai-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**
   ```bash
   # Create .env file in project root
   touch .env
   ```
   
   Add the following environment variables:
   ```env
   ANALYSIS_DB_URI=your_mongodb_uri
   SENTRY_DSN=your_sentry_dsn
   GOOGLE_API_KEY=your_google_api_key
   FIREBASE_PROJECT_ID=your_firebase_project_id
   ```

5. **Firebase setup**
   ```bash
   # Place your Firebase service account JSON file in project root
   cp path/to/firebase-service-account.json ./firebase-service-account.json
   ```

6. **Run the development server**
   ```bash
   python api/index.py
   ```

The server will start on `http://localhost:5000` by default.

---

## 💻 Usage

### **API Endpoints**

#### **Public Endpoints**
- `GET /health` - Health check endpoint
- `GET /api/public/stocks` - Get public stock data

#### **Protected Endpoints**
All protected endpoints require a valid Firebase ID token in the Authorization header:
```
Authorization: Bearer <your-firebase-id-token>
```

- `GET /api/user/portfolio` - Get user's portfolio
- `POST /api/user/analysis` - Request new analysis
- `GET /api/user/history` - Get analysis history

### **Authentication**

Include the Firebase ID token in your requests:
```javascript
const response = await fetch('/api/user/portfolio', {
  headers: {
    'Authorization': `Bearer ${firebaseIdToken}`,
    'Content-Type': 'application/json'
  }
});
```

---

## 🗺️ Roadmap

- [ ] **Enhanced AI Models** - Integration with additional AI providers
- [ ] **Real-time Data Streaming** - WebSocket support for live updates
- [ ] **Advanced Analytics** - Machine learning-based prediction models
- [ ] **API Rate Limiting** - Enhanced rate limiting and quota management
- [ ] **Microservices Architecture** - Split into smaller, focused services
- [ ] **GraphQL Support** - Alternative query interface
- [ ] **Multi-tenant Support** - Enterprise-grade organization features

See the [open issues](https://github.com/your-username/scrybe-ai-backend/issues) for a full list of proposed features and known issues.

---

## 🤝 Contributing

We appreciate your interest in contributing to Scrybe AI Backend! However, please note that this is a **proprietary, closed-source project**. 

### **For Internal Team Members**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Code Standards**

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for any API changes
- Ensure all tests pass before submitting PR

### **External Contributors**

If you're interested in contributing and are not part of the internal team, please reach out to discuss collaboration opportunities.

---

## ⚖️ License

### **Copyright**
© 2025 Scrybe AI. All Rights Reserved.

### **Proprietary License**
This is a proprietary, closed-source project. The code contained within this repository is the sole intellectual property of Scrybe AI. You may not fork, copy, modify, distribute, or use this code in any way without express written permission from the copyright holder.

**Unauthorized use, reproduction, or distribution of this software is strictly prohibited and may result in severe civil and criminal penalties.**

For licensing inquiries, please contact our legal team.

---

## 📞 Contact

**Scrybe AI Development Team**

- **Project Repository:** [https://github.com/your-username/scrybe-ai-backend](https://github.com/your-username/scrybe-ai-backend)
- **Email:** dev@scrybe-ai.com
- **Website:** [https://scrybe-ai.com](https://scrybe-ai.com)
- **Documentation:** [https://docs.scrybe-ai.com](https://docs.scrybe-ai.com)

---

<div align="center">

**Built with ❤️ by the Scrybe AI Team**

[⬆ Back to Top](#scrybe-ai---backend-api)

</div>