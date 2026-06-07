import time
import random
from prometheus_client import start_http_server, Summary, Counter, Gauge, Histogram

# 10 Metriks (Advance kriteria)
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count_total', 'Total request count')
ERROR_COUNT = Counter('error_count_total', 'Total errors')
INFERENCE_TIME = Histogram('inference_duration_seconds', 'Inference duration in seconds')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Data drift detection score')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage_mb', 'Memory usage in MB')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')
PREDICTION_VALUE = Histogram('prediction_value', 'Distribution of predicted values')

@REQUEST_TIME.time()
def process_request(t):
    ACTIVE_REQUESTS.inc()
    time.sleep(t)
    
    # Simulasi inference
    start_time = time.time()
    time.sleep(random.uniform(0.01, 0.1))
    INFERENCE_TIME.observe(time.time() - start_time)
    
    # Simulasi metriks lain
    REQUEST_COUNT.inc()
    
    if random.random() < 0.05:
        ERROR_COUNT.inc()
        
    MODEL_ACCURACY.set(random.uniform(0.85, 0.95))
    DATA_DRIFT_SCORE.set(random.uniform(0.01, 0.2))
    CPU_USAGE.set(random.uniform(10, 80))
    MEMORY_USAGE.set(random.uniform(100, 500))
    PREDICTION_VALUE.observe(random.uniform(0, 1))
    
    ACTIVE_REQUESTS.dec()

if __name__ == '__main__':
    # Start up the server to expose the metrics
    start_http_server(8000)
    print("Prometheus exporter running on port 8000...")
    # Generate some requests
    while True:
        process_request(random.random() * 0.2)
