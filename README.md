# DeteksiKantuk
Proyek ini merupakan sistem deteksi kantuk yang menggunakan parameter fitur wajah dan dataset UTA Real-Life Drowsiness Dataset (UTA RLDD). Dengan memanfaatkan dlib_facelandmark, sistem ini mengekstraksi atribut utama wajah seperti Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), dan Mouth Opening Extent (MOE) untuk mendeteksi tanda-tanda kantuk.

##Fitur Utama
Deteksi Landmark Wajah: Menggunakan dlib untuk melacak titik-titik wajah guna ekstraksi fitur.
Model LSTM: Memanfaatkan Long Short-Term Memory untuk memahami pola temporal pada data video.
Dataset Nyata: Dilatih dan diuji menggunakan UTA RLDD untuk menjamin performa di berbagai skenario nyata.
Indikator Kantuk: Mendeteksi penutupan mata yang lama, menguap, dan perilaku wajah terkait kantuk.

##Panduan Pengaturan Proyek
Ikuti langkah-langkah berikut untuk mengatur dan menjalankan proyek ini.

##Prasyarat
Proyek ini dapat diatur menggunakan berbagai tools Python lainnya. Namun, dalam proyek ini saya menggunakan Conda untuk mengelola lingkungan pengembangan.

##Langkah-langkah :
***Buka Terminal Perangkat Anda***

Aktifkan Environment Conda Anda
Jika Anda menggunakan Conda, ganti "your environment" dengan nama environment Conda Anda, kemudian jalankan perintah berikut:
```bash
conda activate "your environment"
```
***Install library yang dibutuhkan Instal semua library Python yang diperlukan menggunakan file requirement.txt:***
```bash
pip install -r requirement.txt
```
***Jalankan Aplikasi***

python/Demo Program/program.py
