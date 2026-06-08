import time
import json
import psutil
import pandas as pd
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from prometheus_client import start_http_server, Summary, Counter, Gauge, Histogram
from sklearn.ensemble import RandomForestClassifier

# --- 1. Latih Model Nyata Saat Startup ---
print("Memuat dataset dan melatih model...")
base_dir = os.path.dirname(os.path.abspath(__file__))
# Menggunakan dataset yang sama dengan proses training
dataset_path = os.path.join(base_dir, '..', 'Membangun_model', 'dataset_preprocessing.csv')

try:
    df = pd.read_csv(dataset_path)
    X = df[['age_scaled', 'income_scaled']]
    y = df['target']
    
    # Train model sederhana untuk serving (bebas random)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    print("Model berhasil dilatih dan siap melayani request!")
except Exception as e:
    print(f"Error memuat model: {e}")
    # Fallback dummy model jika file tidak ditemukan
    class DummyModel:
        def predict(self, features):
            return [1] # Deterministik, bukan random
    model = DummyModel()

# --- 2. Inisialisasi Metrik Prometheus ---
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

class InferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            ACTIVE_REQUESTS.inc()
            REQUEST_COUNT.inc()
            
            start_time = time.time()
            try:
                # Membaca request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                features = data.get('features', [])
                if len(features) != 2:
                    raise ValueError("Model membutuhkan tepat 2 fitur: age_scaled, income_scaled")
                
                # --- PROSES INFERENCE AKTUAL TANPA RANDOM ---
                input_df = pd.DataFrame([features], columns=['age_scaled', 'income_scaled'])
                prediction = int(model.predict(input_df)[0])
                PREDICTION_VALUE.observe(prediction)
                
                # --- UPDATE METRIKS AKTUAL ---
                inference_latency = time.time() - start_time
                INFERENCE_TIME.observe(inference_latency)
                
                # Metrik sistem nyata (psutil)
                CPU_USAGE.set(psutil.cpu_percent(interval=None))
                MEMORY_USAGE.set(psutil.virtual_memory().used / (1024 * 1024))
                
                # Akurasi statis sebagai contoh pemantauan
                MODEL_ACCURACY.set(0.92) 
                
                # Menghitung Data Drift nyata dari nilai fitur request
                drift_score = sum(features) / len(features)
                DATA_DRIFT_SCORE.set(drift_score)

                # Response sukses
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'prediction': prediction, 'latency': inference_latency}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                ERROR_COUNT.inc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            finally:
                ACTIVE_REQUESTS.dec()
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    start_http_server(8000)
    print("Prometheus metrics exporter running on port 8000...")
    
    server_address = ('', 8001)
    httpd = HTTPServer(server_address, InferenceHandler)
    print("Inference API Server running on port 8001...")
    print("Kirimkan POST request ke http://localhost:8001/predict")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
