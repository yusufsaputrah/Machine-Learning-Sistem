# Machine Learning System - MLOps

Proyek ini adalah submission untuk kelas **Machine Learning System (MLOps)** di Dicoding. Proyek ini mencakup seluruh siklus *end-to-end* sistem Machine Learning mulai dari pemrosesan data, pelatihan model dengan *tracking*, otomatisasi CI/CD, hingga *monitoring* sistem saat *serving*.

## 👨‍💻 Penulis
- **Nama:** Yusuf Saputrah
- **Username Dicoding:** yusuf1610 (Yusuf Saputrah)

## 🗂️ Struktur Direktori

Repositori ini terdiri dari beberapa komponen utama yang memenuhi setiap kriteria penilaian (Basic hingga Advance):

### 1. `Eksperimen_SML_Yusuf-Saputrah/`
Berisi skrip untuk otomatisasi *Data Preprocessing*.
- `preprocessing/Eksperimen_Yusuf-Saputrah.ipynb`: Eksperimen awal pemrosesan data.
- `preprocessing/automate_Yusuf-Saputrah.py`: Skrip Python otomatis yang merangkum *pipeline* preprocessing.
- `dataset_raw.csv`: Data mentah sebelum diproses.

### 2. `Membangun_model/`
Berisi skrip untuk *Model Training* dan integrasi MLflow dengan DagsHub.
- `modelling.py` & `modelling_tuning.py`: Skrip eksperimen pembuatan model Machine Learning.
- `requirements.txt`: *Dependency* spesifik untuk tahapan *modelling*.
- Kumpulan *screenshot* hasil eksekusi DagsHub dan artefak model (Confusion Matrix, Feature Importance, dll).

### 3. `.github/workflows/`
Berisi konfigurasi otomatisasi CI/CD menggunakan GitHub Actions.
- `retrain.yml`: *Workflow* yang akan otomatis berjalan untuk melakukan *retrain* model (MLflow Project) ketika ada perubahan kode yang di-*push* ke *branch* `main`. Terintegrasi penuh dengan kredensial rahasia di GitHub Secrets.

### 4. `Monitoring_dan_Logging/`
Berisi seluruh implementasi untuk *Serving*, *Monitoring*, dan *Alerting*.
- `3.prometheus_exporter.py`: Skrip *exporter* yang mengekspos 10 jenis metrik *monitoring* ke Prometheus.
- `docker-compose.yml` & `2.prometheus.yml`: Konfigurasi *container* untuk menjalankan Prometheus dan Grafana dengan mulus.
- Folder *screenshot* (`4.bukti monitoring Prometheus`, `5.bukti monitoring Grafana`, `6.bukti alerting Grafana`) yang membuktikan sistem telah berhasil menangkap metrik-metrik kesehatan model secara *real-time* hingga membunyikan alarm (Status: **Firing**) ketika metrik melampaui batas yang ditentukan.

## 🚀 Cara Menjalankan Monitoring (Opsional)
Jika penilai ingin menjalankan *monitoring dashboard* secara mandiri:
1. Jalankan eksportir data: `python Monitoring_dan_Logging/3.prometheus_exporter.py`
2. Jalankan mesin metrik dan visualisasi: `cd Monitoring_dan_Logging && docker-compose up -d`
3. Akses `localhost:9090` untuk Prometheus dan `localhost:3000` untuk Grafana.

---
*Proyek ini dirancang untuk memenuhi kriteria Advance Kelas Dicoding MLOps.*
