# 🚀 Machine Learning System - MLOps (Advance Level)

Proyek ini merupakan *submission* untuk kelas **Machine Learning System (MLOps)** di Dicoding. Proyek ini dirancang untuk mencapai **Kriteria Advance** pada seluruh tahapan penilaian, mencakup siklus *end-to-end* mulai dari pemrosesan data, pelatihan model dengan *tracking* (MLflow & DagsHub), otomatisasi CI/CD (GitHub Actions), hingga *monitoring* dan *alerting* sistem saat *serving* (Prometheus & Grafana).

## 👨‍💻 Penulis
- **Nama:** Yusuf Saputrah
- **Username Dicoding:** yusuf1610

## 🔗 Tautan Penting
| Layanan | Tautan |
|---|---|
| **GitHub Repository Utama** | [Machine-Learning-Sistem](https://github.com/yusufsaputrah/Machine-Learning-Sistem) |
| **GitHub Repo Kriteria 1 (Eksperimen SML)** | [Eksperimen_SML](https://github.com/yusufsaputrah/Eksperimen_SML) |
| **GitHub Repo Kriteria 3 (Workflow CI)** | [Workflow-CI](https://github.com/yusufsaputrah/Workflow-CI) |
| **DagsHub Tracking UI** | [yusufsaputrah/Machine-Learning-Sistem](https://dagshub.com/yusufsaputrah/Machine-Learning-Sistem) |
| **Docker Hub Image** | [yusuf1610/sistem-ml-model](https://hub.docker.com/r/yusuf1610/sistem-ml-model) |
| **GitHub Actions CI/CD (Utama)** | [Actions Workflow](https://github.com/yusufsaputrah/Machine-Learning-Sistem/actions) |

---

## 🗂️ Struktur Direktori Submission & Pencapaian Kriteria

Berikut adalah panduan struktur folder di dalam berkas ZIP ini yang merepresentasikan keempat kriteria (Advance):

### 1. Kriteria 1: Eksperimen Dataset (`Eksperimen_SML_Yusuf-Saputrah.txt`)
Karena sudah terintegrasi dengan GitHub Actions, skrip dan repositori khusus untuk preprocessing data dipisah agar lebih rapi.
- Link Repository: [https://github.com/yusufsaputrah/Eksperimen_SML](https://github.com/yusufsaputrah/Eksperimen_SML)
- Tautan repositorinya dapat ditemukan di dalam berkas `Eksperimen_SML_Yusuf-Saputrah.txt`.
- **Pencapaian Advance:** Telah terintegrasi dengan GitHub Actions Workflow untuk melakukan otomatisasi preprocessing.

### 2. Kriteria 2: Membangun Model (`Membangun_model/`)
Berisi skrip untuk *Model Training* dan integrasi MLflow.
- `modelling.py`: Pelatihan model dasar dengan MLflow autologging.
- `modelling_tuning.py`: Hyperparameter tuning (GridSearchCV), pencatatan *manual logging*, dan tambahan artefak pendukung (Confusion Matrix, Feature Importance, Classification Report).
- `Screenshot_dashboard.png`: Bukti *screenshot* dari *dashboard* MLflow di DagsHub yang menampilkan daftar seluruh *runs* dalam eksperimen.
- `Screenshot_artifacts.png`: Bukti *screenshot* halaman *Run Detail / Artifacts* di DagsHub yang menampilkan model, metrics, dan grafik yang tersimpan.
- **Pencapaian Advance:** Eksperimen telah tersimpan secara *online* di DagsHub dan menggunakan *manual logging* dengan lebih dari 2 artefak tambahan.

### 3. Kriteria 3: Workflow CI (`Workflow-CI.txt`)
Implementasi Workflow CI/CD dipisahkan dalam repositori tersendiri sesuai kaidah GitHub Actions.
- Link Repository: [https://github.com/yusufsaputrah/Workflow-CI](https://github.com/yusufsaputrah/Workflow-CI)
- Tautan repositorinya dapat ditemukan di dalam berkas `Workflow-CI.txt`.
- **Pencapaian Advance:** Workflow secara otomatis melakukan *retrain* model, mem-paketkannya menjadi *Docker Image*, lalu melakukan *push* otomatis ke Docker Hub.

### 4. Kriteria 4: Monitoring dan Logging (`Monitor dan Logging/`)
Berisi implementasi *Model Serving*, pemantauan metrik secara *real-time*, dan sistem peringatan.
- `3.prometheus_exporter.py`: Endpoint *FastAPI* nyata yang melayani permintaan (`/predict`) dan mengekspos metrik riil ke Prometheus.
- `7.inference.py`: Skrip untuk melakukan simulasi *request inference* ke server API.
- Folder Bukti (`4.bukti monitoring Prometheus`, `5.bukti monitoring Grafana`, `6.bukti alerting Grafana`): Berisi bukti-bukti *screenshot* untuk penilaian.
- **Pencapaian Advance:** Model di-*serve* secara nyata, pemantauan menggunakan **lebih dari 10 metrik** riil, dan memiliki **3 aturan *alerting*** di Grafana.

---

## 🚀 Cara Menjalankan Monitoring Secara Lokal

Jika ingin menjalankan *server inference* dan *monitoring dashboard* secara mandiri:
1. **Jalankan API Server FastAPI:** 
   ```bash
   python3 "Monitor dan Logging/3.prometheus_exporter.py"
   ```
2. **Jalankan Prometheus & Grafana (Docker):** 
   ```bash
   cd "Monitor dan Logging"
   docker-compose up -d
   ```
3. **Simulasikan Traffic Secara Aktif:**
   ```bash
   python3 "Monitor dan Logging/7.inference.py"
   ```

---
*Dibuat untuk memenuhi standar kompetensi kelulusan dengan predikat terbaik di kelas Machine Learning System (MLOps) Dicoding.*
