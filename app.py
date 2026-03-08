import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go # Untuk membuat grafik Gauge

# 1. Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(
    page_title="Sistem Deteksi Pneumonia Pediatrik",
    page_icon="🏥",
    layout="wide", # Menggunakan lebar layar penuh
    initial_sidebar_state="expanded"
)

# 2. Load Model dengan Caching agar Cepat
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_pneumonia_resnet50.keras')

model = load_my_model()

# ------------------------------------------------------------------
# 3. SIDEBAR: Profil dan Cara Penggunaan
# ------------------------------------------------------------------
with st.sidebar:
    left_co, cent_co, last_co = st.columns([1, 2, 1])
    with cent_co:
        st.image("Logo Gundar.png", width=120)
    st.markdown("""
        <div style="text-align: center;">
            <h2 style="margin-bottom: 0;">Skripsi Sistem Informasi</h2>
            <h3 style="margin-top: 0; margin-bottom: 0;">Muhammad Yogi Prasojo</h3>
            <p style="font-size: 18px; margin-top: 0;"><b>11122016</b></p>
            <p style="font-size: 16px;">Universitas Gunadarma</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("---")

    st.header("📖 Cara Penggunaan")
    st.info(
        "1. Siapkan foto Chest X-Ray pasien anak (Usia 1-5 tahun).\n"
        "2. Pastikan format foto JPG, JPEG, atau PNG.\n"
        "3. Unggah foto pada panel di sebelah kanan.\n"
        "4. Tunggu AI selesai menganalisis.\n"
        "5. Hasil diagnosa dan tingkat keyakinan akan muncul."
    )
    
    st.warning("**Disclaimer:** Ini adalah prototipe penelitian dan bukan diagnosa medis final.")

# ------------------------------------------------------------------
# 4. MAIN AREA: Judul dan Fitur Unggah
# ------------------------------------------------------------------
col_title, col_icon = st.columns([8, 1])
with col_title:
    st.title("🏥 Sistem Cerdas Deteksi Dini Pneumonia pada Pasien Pediatrik")
with col_icon:
    st.write("") # Kosong untuk menyamakan tinggi emoji

st.write("### Implementasi Deep Learning ResNet50 untuk Pendukung Keputusan Klinis")
st.write("---")

# Area Unggah
st.header("📁 Unggah Foto X-Ray")
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------------
# 5. PEMROSESAN DAN TAMPILAN HASIL (Hanya muncul jika file diunggah)
# ------------------------------------------------------------------
if uploaded_file is not None:
    # Menggunakan columns untuk membagi tampilan (Gambar vs Hasil)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Gambar yang Dianalisis")
        image = Image.open(uploaded_file).convert('RGB')
        
        # Preprocessing Gambar
        img_input = image.resize((224, 224))
        img_array = np.array(img_input) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Menampilkan gambar dengan border warna dasar
        st.image(image, caption='Chest X-Ray', use_column_width=True)

    with col2:
        st.subheader("📊 Hasil Analisis AI")
        
        with st.spinner('Sedang menganalisis struktur paru-paru...'):
            prediction = model.predict(img_array)
            probabilitas_pneu = prediction[0][0]
            
        # Perhitungan Persentase
        if probabilitas_pneu > 0.5:
            diagnosa = "PNEUMONIA"
            persentase = probabilitas_pneu * 100
            pesan = st.error # Menggunakan kotak warna Merah
            warna_gauge = "#FF4136" # Merah
        else:
            diagnosa = "NORMAL"
            persentase = (1 - probabilitas_pneu) * 100
            pesan = st.success # Menggunakan kotak warna Hijau
            warna_gauge = "#2ECC40" # Hijau

        # --- Tampilan Hasil Diagnosa ---
        pesan(f"## Diagnosa: **{diagnosa}**")
        
        # --- Membuat Grafik Gauge (Speedometer) dengan Plotly ---
        fig_gauge = go.Figure(go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=persentase,
            mode="gauge+number",
            title={'text': "Tingkat Keyakinan AI (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': warna_gauge},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': warna_gauge}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        # Menampilkan Grafik
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_column_width=True)

        # --- Bagian Prescriptive (Saran) ---
        st.markdown("### 📋 Rekomendasi/Tindakan:")
        if diagnosa == "PNEUMONIA":
            st.markdown(
                "- ⚠️ **Segera konsultasikan dengan Dokter Spesialis Anak.**\n"
                "- Lakukan pemeriksaan laboratorium lengkap (misal: cek darah).\n"
                "- Pertimbangkan rawat inap jika terdapat sesak napas berat."
            )
        else:
            st.markdown(
                "- ✅ **Paru-paru terlihat bersih dan normal.**\n"
                "- Tetap jaga kebersihan lingkungan dan pola makan anak.\n"
                "- Jika gejala klinis (batuk/demam) berlanjut, hubungi dokter."
            )

st.write("---")
# Footer sederhana
st.markdown("<div style='text-align: center;'>© 2026 - Skripsi Sistem Informasi Universitas Gunadarma</div>", unsafe_allow_stdio=True)