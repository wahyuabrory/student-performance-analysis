# Sistem Prediksi Dropout Mahasiswa

## Pemahaman Bisnis
Perguruan tinggi sering menghadapi masalah putus kuliah (dropout) yang berdampak negatif pada beberapa aspek, termasuk reputasi institusi, efisiensi pendanaan, dan target kelulusan. Sistem prediksi dropout mahasiswa ini dikembangkan untuk membantu institusi pendidikan dalam mengidentifikasi mahasiswa yang berisiko tinggi dropout sejak dini, sehingga intervensi tepat dapat diberikan untuk mengurangi tingkat dropout.

### Permasalahan Bisnis
1. **Tingginya Tingkat Dropout**: Banyak institusi pendidikan mengalami tingkat dropout yang signifikan, yang berpengaruh pada metrik performa institusi.
2. **Keterlambatan Identifikasi**: Seringkali mahasiswa yang berisiko dropout teridentifikasi terlambat, saat intervensi sudah kurang efektif.
3. **Ketidakefisienan Alokasi Sumber Daya**: Ketidakmampuan untuk mengidentifikasi mahasiswa berisiko menyebabkan alokasi sumber daya bantuan yang tidak tepat sasaran.
4. **Kurangnya Pendekatan Personalisasi**: Institusi sering kekurangan tools untuk merancang intervensi yang dipersonalisasi berdasarkan faktor risiko spesifik tiap mahasiswa.

### Cakupan Proyek
Proyek ini mencakup:
1. Pengembangan model machine learning untuk memprediksi kemungkinan mahasiswa mengalami dropout
2. Implementasi aplikasi web interaktif untuk memudahkan penggunaan model oleh staf akademik
3. Fitur pembuatan rencana intervensi yang dipersonalisasi berdasarkan faktor risiko teridentifikasi
4. Visualisasi hasil prediksi dan rekomendasi tindakan

### Persiapan

**Sumber data**: Dataset berisi informasi akademik dan demografis mahasiswa, termasuk nilai akademik, kehadiran, latar belakang keluarga, dan status keuangan. Data telah diproses dan disimpan dalam format CSV.

**Setup environment**:
```bash
# Clone repositori
git clone https://github.com/username/student-performance-analysis.git
cd student-performance-analysis

# Instalasi dependensi
pip install -r requirements.txt

# Menjalankan aplikasi
streamlit run app.py
```

## Dashboard Bisnis

## Menjalankan Sistem Machine Learning
Sistem prediksi dropout mahasiswa dapat dijalankan dengan langkah-langkah berikut:

```bash
# Menjalankan aplikasi prediksi
streamlit run app.py
```

Setelah aplikasi berjalan:
1. Isi formulir dengan data mahasiswa yang ingin dianalisis
2. Klik tombol "Predict" untuk melihat hasil prediksi
3. Jika mahasiswa terdeteksi berisiko dropout, klik tombol "Generate Intervention Plan" untuk mendapatkan rencana intervensi terperinci
4. Gunakan informasi ini untuk menerapkan strategi yang tepat dalam membantu mahasiswa

## Kesimpulan
Sistem prediksi dropout mahasiswa ini berhasil mengidentifikasi mahasiswa berisiko dengan akurasi tinggi. Model machine learning yang dikembangkan dapat mengenali pola dan faktor risiko yang mungkin terlewatkan oleh pengamatan manual. Dengan menggunakan sistem ini, institusi pendidikan dapat:

1. Mengidentifikasi mahasiswa berisiko sejak dini
2. Memahami faktor-faktor utama yang berkontribusi pada risiko dropout
3. Menerapkan intervensi tepat waktu dan dipersonalisasi
4. Meningkatkan tingkat retensi dan keberhasilan mahasiswa

### Rekomendasi Tindakan
Berdasarkan analisis dan sistem yang dikembangkan, berikut beberapa rekomendasi tindakan:

- **Pemantauan Berkelanjutan**: Implementasikan sistem pemantauan berkelanjutan untuk mahasiswa berisiko tinggi, dengan evaluasi ulang setiap semester
- **Program Mentoring Terstruktur**: Kembangkan program mentoring akademik yang dipersonalisasi berdasarkan kebutuhan spesifik mahasiswa
- **Dukungan Finansial Terarah**: Alokasikan sumber daya finansial untuk mahasiswa yang mengalami kesulitan keuangan yang berpotensi menyebabkan dropout
- **Peningkatan Keterlibatan**: Dorong partisipasi mahasiswa dalam kegiatan kampus untuk meningkatkan rasa memiliki dan keterikatan dengan institusi
- **Pembimbing Akademik Proaktif**: Latih pembimbing akademik untuk secara proaktif mendekati mahasiswa dengan tanda-tanda peringatan dini
- **Workshop Keterampilan Belajar**: Adakan workshop untuk membantu mahasiswa mengembangkan keterampilan belajar, manajemen waktu, dan strategi ujian

## Fitur Tambahan
- **Prediksi Real-time**: Kemampuan melakukan prediksi secara real-time berdasarkan data terbaru
- **Rencana Intervensi Terperinci**: Generasi otomatis rencana intervensi yang disesuaikan dengan profil risiko mahasiswa
- **Analisis Faktor Risiko**: Visualisasi faktor-faktor yang paling berkontribusi pada risiko dropout
- **Timeline Implementasi**: Jadwal rekomendasi tindakan dengan prioritas jangka pendek hingga jangka panjang

## Pengembang
Moh. Wahyu Abrory - [LinkedIn](http://linkedin.com/in/wahyuabrory)
*Copyright © 2025 - Hak Cipta Dilindungi*
