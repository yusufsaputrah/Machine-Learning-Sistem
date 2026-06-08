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

## 🗂️ Struktur Direktori & Pencapaian Kriteria

Repositori ini terdiri dari 4 komponen utama yang masing-masing memenuhi **Kriteria Advance**:

### 1. Kriteria 1: Eksperimen Dataset (`Eksperimen_SML_Yusuf-Saputrah/`)
> **Catatan untuk Reviewer:** Halo kak, sesuai feedback sebelumnya, Kriteria 1 ini sudah saya pisahkan ke repository sendiri ya.
> Link Repository: [https://github.com/yusufsaputrah/Eksperimen_SML](https://github.com/yusufsaputrah/Eksperimen_SML)

Berisi skrip untuk eksperimen dan otomatisasi *Data Preprocessing*.
- `preprocessing/Eksperimen_Yusuf-Saputrah.ipynb`: Eksperimen awal (Data Loading, EDA, Preprocessing).
- `preprocessing/automate_Yusuf-Saputrah.py`: Skrip *pipeline* preprocessing otomatis.
- **Pencapaian Advance:** Telah terintegrasi dengan GitHub Actions Workflow untuk melakukan otomatisasi preprocessing.

### 2. Kriteria 2: Membangun Model (`Membangun_model/`)
Berisi skrip untuk *Model Training* dan integrasi MLflow.
- `modelling.py`: Pelatihan model dasar dengan MLflow autologging.
- `modelling_tuning.py`: Hyperparameter tuning (GridSearchCV), pencatatan *manual logging*, dan tambahan artefak pendukung (Confusion Matrix, Feature Importance, Classification Report).
- **Pencapaian Advance:** Eksperimen telah tersimpan secara *online* di DagsHub dan menggunakan *manual logging* dengan lebih dari 2 artefak tambahan.

### 3. Kriteria 3: Workflow CI (`Workflow-CI/`)
> **Catatan untuk Reviewer:** Halo kak, sesuai feedback sebelumnya, Kriteria 3 ini juga sudah saya pisahkan ke repository sendiri ya.
> Link Repository: [https://github.com/yusufsaputrah/Workflow-CI](https://github.com/yusufsaputrah/Workflow-CI)

Berisi konfigurasi otomatisasi CI/CD untuk pelatihan ulang model.
- `MLProject/`: Folder utama MLflow Project (berisi skrip training dan `conda.yaml`).
- `.github/workflows/retrain.yml`: *Workflow* GitHub Actions untuk otomatisasi *retrain*.
- **Pencapaian Advance:** Workflow secara otomatis melakukan *retrain* model, mem-paketkannya menjadi *Docker Image*, lalu melakukan *push* otomatis ke Docker Hub.

### 4. Kriteria 4: Monitoring dan Logging (`Monitoring_dan_Logging/`)
Berisi implementasi *Model Serving*, pemantauan metrik secara *real-time*, dan sistem peringatan.
- `3.prometheus_exporter.py`: Server *inference* nyata yang melayani permintaan (`/predict`) dan mengekspos metrik kesehatan model ke Prometheus.
- `7.inference.py`: Skrip untuk melakukan simulasi *request inference* ke server.
- `docker-compose.yml` & `2.prometheus.yml`: Orkestrasi kontainer Prometheus dan Grafana.
- Folder Bukti: Berisi *screenshot* dari Docker Desktop, 10+ metrik Prometheus/Grafana, dan konfigurasi peringatan (*alerting*).
- **Pencapaian Advance:** Model di-*serve* secara nyata (termasuk via Docker), pemantauan menggunakan **lebih dari 10 metrik**, dan memiliki **3 aturan *alerting*** di Grafana.

---

## 🚀 Cara Menjalankan Monitoring Secara Lokal

Jika ingin menjalankan *server inference* dan *monitoring dashboard* secara mandiri:
1. **Jalankan Inference Server & Exporter:** 
   ```bash
   python3 Monitoring_dan_Logging/3.prometheus_exporter.py
   ```
2. **Jalankan Prometheus & Grafana:** 
   ```bash
   cd Monitoring_dan_Logging
   docker-compose up -d
   ```
3. **Simulasi Traffic (Opsional):**
   ```bash
   python3 Monitoring_dan_Logging/7.inference.py
   ```
4. **Akses Dashboard:**
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000`

---
*Dibuat untuk memenuhi standar kompetensi kelulusan dengan predikat terbaik di kelas Machine Learning System (MLOps) Dicoding.*
