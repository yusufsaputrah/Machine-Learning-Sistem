import requests
import time
import random

def simulate_traffic():
    print("Simulating traffic to the model endpoint/exporter...")
    # Dalam skenario nyata, ini akan mengirim request ke endpoint FastAPI / MLflow Serve
    # Di sini kita anggap exporter sudah berjalan dan kita hanya mensimulasikan waktu berjalannya sistem
    for i in range(50):
        print(f"Simulated request {i+1}...")
        time.sleep(random.uniform(0.1, 1.0))
        
if __name__ == "__main__":
    simulate_traffic()
    print("Inference simulation finished.")
