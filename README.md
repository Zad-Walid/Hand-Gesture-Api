# Hand-Gesture-Api

## Overview
This project provides a FastAPI-based REST API for serving a hand gesture classification model. The API predicts hand gestures from normalized landmark data and is designed for easy integration and deployment.

## Features
- FastAPI REST API for real-time hand gesture prediction
- Model and label encoder loading for production inference
- Logging and error handling
- Docker and Docker Compose support for easy deployment
- Prometheus and Grafana integration for monitoring
- CI/CD pipeline with GitHub Actions for automated testing and deployment

## Monitoring & System Metrics
The API is instrumented with [Prometheus FastAPI Instrumentator](https://github.com/trallard/prometheus-fastapi-instrumentator) for real-time monitoring of system and application metrics.  
A Prometheus service is included in the `docker-compose.yml` for scraping metrics, and Grafana is used for dashboard visualization.

**How to access metrics:**
- Prometheus metrics endpoint: `http://localhost:8000/metrics`
- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000`

**Monitored metrics include:**
- Model related: Inference Latency	
- Data related: Average Request Duration
- Server related: CPU Usage

## Dashboard Screenshot

Below are example screenshots of the Grafana dashboard monitoring the API:

![inference_latency](https://github.com/user-attachments/assets/975cbaeb-228a-4481-9de8-3780806fd48e)  
This metric measures the time taken by the ML model to return a prediction.  
High latency may signal inefficiency or the need for optimization.

![avg_req_dur](https://github.com/user-attachments/assets/a892b0be-eff0-473f-b439-e342c4157036)  
Represents how long it takes on average to respond to incoming requests.  
Tracks data flow latency and user experience.

![cpu_usage](https://github.com/user-attachments/assets/91cedaf2-5086-488c-8e0b-dd39a297013d)  
Indicates how much CPU the containerized API is consuming.  
Helps detect resource bottlenecks.

---

## Project Structure

```
Hand-Gesture-Api/
├── api/
│   ├── main.py
│   ├── output/
│   │   ├── model.pkl
│   │   └── label_encoder.joblib
│   └── logs/
│       └── api.log
├── dataset/
│   └── hand_landmarks_data.csv
├── monitoring/
│   └── prometheus.yml
├── src/
│   └── train.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── actions.yml
```

---

## Quick Start

1. **Build and run with Docker Compose:**
    ```sh
    docker-compose up --build
    ```

2. **API Endpoints:**
    - `GET /` — Health check
    - `GET /health` — Service status
    - `POST /predict` — Predict hand gesture from landmarks

3. **Monitoring:**
    - Prometheus: [http://localhost:9090](http://localhost:9090)
    - Grafana: [http://localhost:3000](http://localhost:3000)

---

## CI/CD

This project uses GitHub Actions for continuous integration and deployment.  
On every push to the `main` branch:
- Tests are run automatically.
- If successful, the latest code is deployed to the production server via SSH and Docker Compose.

---

## License
MIT