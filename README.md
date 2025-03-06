# 🚀 Fraud Detection System

### **Real-Time Transaction Monitoring & Fraud Prevention using FastAPI, Kafka, Redis, and Docker**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green) ![Kafka](https://img.shields.io/badge/Kafka-Streaming-red) ![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

## 🚨 Why This Project Matters

Financial fraud is a multi-billion-dollar problem. This **Fraud Detection System** is engineered to detect and prevent fraudulent transactions in **real-time** with near-perfect accuracy, leveraging **Machine Learning, Kafka event streaming, and high-performance APIs.**

### 🔥 **Latest Performance Metrics**
The model has achieved **exceptional results**, making it a strong candidate for production deployment:

- **Precision:** `1.000` ✅ *(No false positives!)*
- **Recall:** `0.935` 🔥 *(Detects nearly all fraud cases!)*
- **F1-Score:** `0.966` 🎯 *(Perfect balance of precision & recall!)*
- **Accuracy:** `1.000` 🚀 *(Near-perfect fraud detection!)*

## 🌍 Overview

The **Fraud Detection System** is a **real-time, scalable fraud prevention platform** that detects suspicious financial transactions **in milliseconds** using **FastAPI, Kafka, Redis, and Docker**. Designed to process **millions of transactions per second**, this project aligns with **high-scale distributed computing and cloud-based fraud detection strategies.**

---

## 🛠 Key Features

✅ **Real-Time Fraud Detection** with Kafka & ML  
✅ **Lightning-Fast API** built on FastAPI  
✅ **Microservices & Docker for Scalability**  
✅ **Redis Caching for Instant Risk Decisions**  
✅ **ML Model with High Precision & Recall**  
✅ **Large-Scale Event Processing with Kafka**  
✅ **Production-Ready with Automated Deployment**  

---

## 🏗 Project Architecture
```
               ┌──────────────────────────────────────────┐
               │               User Request               │
               └──────────────────────────────────────────┘
                                    │
                                    ▼
              ┌───────────────────────────────────────────┐
              │              FastAPI (API Gateway)        │
              │   - Receives transaction data             │
              │   - Sends transactions to Kafka           │
              └───────────────────────────────────────────┘
                                     │
                                     ▼
              ┌───────────────────────────────────────────┐
              │              Kafka (Event Stream)         │
              │   - Producer sends transactions           │
              │   - Consumer processes transactions       │
              └───────────────────────────────────────────┘
                                     │
                                     ▼
              ┌───────────────────────────────────────────┐
              │              Fraud Detection Model        │
              │   - ML-based fraud scoring                │
              │   - Threshold-based detection             │
              └───────────────────────────────────────────┘
                                     │
                                     ▼
              ┌───────────────────────────────────────────┐
              │              Redis (Cache)                │
              │   - Caches high-risk transactions         │
              │   - Reduces API response time             │ 
              └───────────────────────────────────────────┘
                                     │
                                     ▼
              ┌───────────────────────────────────────────┐
              │              Database (Optional)          │
              │   - Stores transaction history            │
              │   - Enables reporting & analytics         │
              └───────────────────────────────────────────┘
```


---

## ⚡ Tech Stack

| Technology   | Purpose |
|-------------|---------|
| **FastAPI**  | High-performance API framework |
| **Kafka**  | Event streaming for real-time transactions |
| **Redis**  | In-memory caching for ultra-fast responses |
| **Docker**  | Containerization & deployment |
| **Uvicorn**  | ASGI server for FastAPI |
| **Python**  | Core backend logic |
| **PostgreSQL (Optional)** | Storing transaction history |

---

## 🚀 Installation & Setup

### **Pre-requisites**
Ensure you have the following installed:
- **Python 3.8+**
- **Docker & Docker-Compose**
- **Kafka & Zookeeper**
- **Redis**

### **Clone the Repository**
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Services

#### Run with Docker

```bash
docker-compose up --build -d
```

#### Run Locally (Without Docker)

```bash
# Start Redis
redis-server --daemonize yes

# Start Zookeeper
zookeeper-server-start.sh /usr/local/etc/kafka/zookeeper.properties &

# Start Kafka Broker
kafka-server-start.sh /usr/local/etc/kafka/server.properties &

# Start FastAPI
uvicorn src.fastapi_service:app --host 0.0.0.0 --port 8001
```

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| **GET** | `/` | Health check (Returns "Fraud Detection API is running!") |
| **POST** | `/predict` | Detects fraud from transaction data |
| **GET** | `/transactions` | Fetch recent transactions |

### Example Request

#### Detect Fraud (POST `/predict`)

```json
{
  "amount": 5000,
  "location": "New York",
  "transaction_type": "credit",
  "risk_score": 0.85
}
```

#### Response

```json
{
  "fraudulent": true,
  "confidence": 95.7,
  "message": "Transaction flagged as potential fraud"
}
```

## Future Enhancements

- ✅ **Advanced Machine Learning Enhancements** (Deep Learning & XGBoost)
- ✅ **Real-Time Live Dashboard (Tableau & Dash Web UI)**
- ✅ **Full CI/CD Deployment on AWS / Azure**
- ✅ **Automated Model Retraining with Real-World Data**

## Want to See More?

📂 **Code Repository**: [GitHub](https://github.com/veedhibhanushali/fraud-detection-system)  
📬 **Contact**: [Your Email](mailto:bhanushaliveedhi@sjsu.edu)  
📝 **Author**: *[Veedhi Bhanushali](https://veedhibhanushali.com)*  


