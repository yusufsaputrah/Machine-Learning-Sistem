import requests
import time
import random

def simulate_traffic():
    print("Mengirimkan real HTTP requests ke endpoint Model Serving API...")
    url = "http://localhost:8000/predict"
    
    for i in range(50):
        print(f"Request ke-{i+1}...")
        
        # Data payload nyata dengan 2 fitur (age_scaled, income_scaled)
        # Sesuai dengan dataset yang dilatih model
        age_scaled = random.uniform(-2.0, 2.0)
        income_scaled = random.uniform(-2.0, 2.0)
        
        payload = {
            "features": [age_scaled, income_scaled]
        }
        
        try:
            start_req = time.time()
            response = requests.post(url, json=payload, timeout=2)
            latency = time.time() - start_req
            
            if response.status_code == 200:
                print(f" Sukses! Fitur: [{age_scaled:.2f}, {income_scaled:.2f}] | Prediksi Kelas: {response.json().get('prediction')} | Latency: {latency:.4f}s")
            else:
                print(f" Gagal dengan status {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f" Error koneksi: {e}")
            
        time.sleep(0.5) # Jeda antar request nyata
        
if __name__ == "__main__":
    simulate_traffic()
    print("Selesai mengirimkan traffic.")
