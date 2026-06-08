import requests
import time
import random

def simulate_traffic():
    print("Mengirimkan real HTTP requests ke endpoint Model Serving API...")
    url = "http://localhost:8001/predict"
    
    for i in range(50):
        print(f"Request ke-{i+1}...")
        
        # Data payload nyata yang akan mempengaruhi metrik drift di server
        payload = {
            "features": [random.uniform(0, 1) for _ in range(5)]
        }
        
        try:
            start_req = time.time()
            response = requests.post(url, json=payload, timeout=2)
            latency = time.time() - start_req
            
            if response.status_code == 200:
                print(f" Sukses! Prediksi: {response.json().get('prediction'):.4f} | Latency Jaringan: {latency:.4f}s")
            else:
                print(f" Gagal dengan status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f" Error koneksi: {e}")
            
        time.sleep(random.uniform(0.5, 2.0)) # Jeda antar request nyata
        
if __name__ == "__main__":
    simulate_traffic()
    print("Selesai mengirimkan traffic.")
