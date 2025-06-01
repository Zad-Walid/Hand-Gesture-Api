# Hand-Gesture-Api

## Overview
This project provides a FastAPI-based REST API for serving a hand gesture classification model. The API predicts hand gestures from normalized landmark data and is designed for easy integration and deployment.

## Features
- FastAPI REST API for real-time hand gesture prediction
- Model and label encoder loading for production inference
- Logging and error handling
- Docker and Docker Compose support for easy deployment

## Monitoring & System Metrics
The API is instrumented with [Prometheus FastAPI Instrumentator](https://github.com/trallard/prometheus-fastapi-instrumentator) for real-time monitoring of system and application metrics.  
A Prometheus service is included in the `docker-compose.yml` for scraping metrics, and Grafana is used for dashboard visualization.

**How to access metrics:**
- Prometheus metrics endpoint: `http://localhost:8000/metrics`
- Prometheus UI: `http://localhost:9090`
- Grafana UI: `http://localhost:3000`

**Monitored metrics include:**
- Model related : Inference Latency	
- Data related : Average Request Duration
- Server related : CPU Usage

## Dashboard Screenshot

Below is an example screenshot of the Grafana dashboard monitoring the API:

![Grafana Dashboard Example](monitoring/dashboard_screenshot.png)
This metric measures the time taken by the ML model to return a prediction.
High latency may signal inefficiency or the need for optimization.

![Grafana Dashboard Example](monitoring/dashboard_screenshot.png)
Represents how long it takes on average to respond to incoming requests. 
Tracks data flow latency and user experience.

![Grafana Dashboard Example](monitoring/dashboard_screenshot.png)
Indicates how much CPU the containerized API is consuming.
Helps detect resource bottlenecks.

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

## License
MIT