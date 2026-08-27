import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wavfile
import scipy.signal as signal_lib
import io

# --- הגדרת תצורת העמוד ---
st.set_page_config(
    page_title="Advanced DSP Workstation Ultimate",
    page_icon="🎛️",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #161b22; }
    </style>
""", unsafe_allow_html=True)

st.title("🎛️ Advanced DSP Workstation Ultimate")
st.write("תחנת עבודה מתקדמת לעיבוד אותות ספרתי, ניתוח פאזה, דיסטורשן, פילטור וספקטרום.")

# --- יצירת כרטיסיות ראשיות ---
tab_analyzer, tab_generator, tab_waterfall = st.tabs([
    "📊 1. ספקטרום אנלייזר + פאזה ודיסטורשן", 
    "📈 2. מחולל אותות", 
    "🌊 3. תצוגת מפל מים (Waterfall)"
])

# ==========================================
# כרטיסייה 1: ספקטרום אנלייזר + פאזה ודיסטורשן
# ==========================================
with tab_analyzer:
    st.header("מנתח ספקטרום מתקדם עם תגובת פאזה, דיסטורשן וסינון")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        analyzer_source = st.selectbox(
            "בחר מקור אות לניתוח:",
            ["גלים סינתטיים מורכבים", "גלים לא-הרמוניים (ריבועי/שן-מסור)", "רעש סטטיסטי וסביבתי", "הקלטת מיקרופון חיה", "העלאת קובץ נתונים/שמע חיצוני"],
            key="ana_src"
        )
        window_type = st.selectbox("סוג חלון ל-FFT:", ["Hann", "Hamming", "Blackman", "Rectangular (ללא)"], key="win_t")

    with col_c2:
        sr_ana = st.slider("קצב דגימה (Hz)", 8000, 48000, 44100, 1000, key="sr_a")
        duration_ana = st.slider("משך חלון (שניות)", 0.1, 3.0, 1.0, 0.1, key="dur_a")
        noise_boost = st.slider("הזרקת רעש לבן (SNR)", 0.0, 1.0, 0.0, 0.05, key="noise_b")

    with col_c3:
        st.subheader("⚙️ עיוותים, פילטרים ופאזה")
        distortion_drive = st.slider("עיוות לא-ליניארי / Clipping (Drive)", 1.0, 10.0, 1.0, 0.5)
        apply_filter = st.checkbox("הפעל מסנן (Filter)")
        filter_type = st.selectbox("סוג מסנן:", ["Low-Pass (מעביר-נמוכים)", "High-Pass (מעביר-גבוהים)", "Band-Pass (מעביר-רצועה)"])
        
        if "Band-Pass" in filter_type:
            c_low = st.slider("תדר חיתוך תחתון (Hz)", 100, 5000, 500, 50)
            c_high = st.slider("תדר חיתוך עליון (Hz)", 1000, 20000, 4000, 100)
        else:
            cutoff_freq = st.slider("תדר חיתוך (Cutoff Hz)", 100, 10000, 1500, 100)
        
        show_phase = st.checkbox("הצג ספקטרום פאזה (Phase Response)")

    num_samples_ana = int(sr_ana * duration_ana)
    t_ana = np.linspace(0, duration_ana, num_samples_ana, endpoint=False)
    signal_ana = None
    info_ana = ""

    # הפקת האות
    if analyzer_source == "גלים סינתטיים מורכבים":
        f1 = st.slider("תדר 1 (Hz)", 20, 5000, 440, key="f1_a")
        a1 = st.slider("אמפליטודה 1", 0.1, 2.0, 1.0, key="a1_a")
        f2 = st.slider("תדר 2 (Hz)", 20, 10000, 2500, key="f2_a")
        a2 = st.slider("אמפליטודה 2", 0.1, 2.0, 0.8, key="a2_a")
        signal_ana = a1 * np.sin(2 * np.pi * f1 * t_ana) + a2 * np.sin(2 * np.pi * f2 * t_ana)
        info_ana = f"סינוסים: {f1}Hz ו-{f2}Hz"

    elif analyzer_source == "גלים לא-הרמוניים (ריבועי/שן-מסור)":
        wave_type = st.radio("סוג הגל:", ["גל ריבועי (Square)", "גל שן-מסור (Sawtooth)"], key="wt_a")
        base_freq = st.slider("תדר בסיס (Hz)", 20, 2000, 220, key="bf_a")
        if "ריבועי" in wave_type:
            signal_ana = np.sign(np.sin(2 * np.pi * base_freq * t_ana))
            info_ana = f"גל ריבועי ב- {base_freq}Hz"
        else:
            signal_ana = 2 * (t_ana * base_freq - np.floor(0.5 + t_ana * base_freq))
            info_ana = f"גל שן-מסור ב- {base_freq}Hz"

    elif analyzer_source == "רעש סטטיסטי וסביבתי":
        noise_type = st.radio("סוג רעש:", ["רעש לבן (White Noise)", "רעש בראוני/אדום (Brownian Noise)"], key="nt_a")
        if "לבן" in noise_type:
            signal_ana = np.random.normal(0, 1, num_samples_ana)
            info_ana = "רעש לבן אקראי"
        else:
            white = np.random.normal(0, 1, num_samples_ana)
            signal_ana = np.cumsum(white)
            signal_ana = signal_ana / np.max(np.abs(signal_ana))
            info_ana = "רעש בראוני"

    elif analyzer_source == "הקלטת מיקרופון חיה":
        if st.button("🔴 התחל הקלטה חיה", key="mic_a"):
            with st.spinner(f"מקליט למשך {duration_ana} שניות..."):
                audio_rec = sd.rec(int(duration_ana * sr_ana), samplerate=sr_ana, channels=1, blocking=True)
                signal_ana = audio_rec[:, 0]
                info_ana = "הקלטת מיקרופון חיה"

    elif analyzer_source == "העלאת קובץ נתונים/שמע חיצוני":
        uploaded_file = st.file_uploader("העלה קובץ שמע (.wav) או קובץ טקסט/CSV", type=["wav", "csv", "txt"], key="up_a")
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.wav'):
                sr_file, signal_ana = wavfile.read(uploaded_file)
                if len(signal_ana.shape) > 1: signal_ana = signal_ana[:, 0]
                sr_ana = sr_file
                info_ana = f"קובץ שמע: {uploaded_file.name}"
            else:
                data = np.loadtxt(uploaded_file, delimiter=",")
                signal_ana = data[:, 0] if len(data.shape) > 1 else data
                info_ana = f"קובץ נתונים: {uploaded_file.name}"

    if signal_ana is not None and len(signal_ana) > 0:
        # 1. הזרקת רעש
        if noise_boost > 0:
            signal_ana = signal_ana + noise_boost * np.random.normal(0, 1, len(signal_ana))
            info_ana += f" + רעש ({noise_boost})"

        # 2. עיוות לא-ליניארי (Distortion / Clipping)
        if distortion_drive > 1.0:
            signal_ana = np.clip(signal_ana * distortion_drive, -1.0, 1.0)
            info_ana += f" | דיסטורשן (Drive: {distortion_drive}x)"

        # 3. פילטר ספרתי
        if apply_filter:
            nyq = sr_ana / 2
            if "Band-Pass" in filter_type:
                low_norm = max(0.01, min(0.98, c_low / nyq))
                high_norm = max(low_norm + 0.01, min(0.99, c_high / nyq))
                b, a = signal_lib.butter(4, [low_norm, high_norm], btype='band')
                info_filter = f"Band-Pass ({c_low}-{c_high}Hz)"
            else:
                cutoff_norm = max(0.01, min(0.99, cutoff_freq / nyq))
                b, a = signal_lib.butter(4, cutoff_norm, btype='low' if 'Low' in filter_type else 'high')
                info_filter = f"{filter_type}, חיתוך: {cutoff_freq}Hz"
            
            signal_to_analyze = signal_lib.lfilter(b, a, signal_ana)
            info_ana += f" | מסונן ({info_filter})"
        else:
            signal_to_analyze = signal_ana

        st.success(f"מנתח את האות: **{info_ana}**")
        
        # בחירת חלון FFT וחישוב
        if "Hann" in window_type: window = np.hanning(len(signal_to_analyze))
        elif "Hamming" in window_type: window = np.hamming(len(signal_to_analyze))
        elif "Blackman" in window_type: window = np.blackman(len(signal_to_analyze))
        else: window = np.ones(len(signal_to_analyze))

        windowed = signal_to_analyze * window
        fft_complex = np.fft.rfft(windowed)
        fft_mag = np.abs(fft_complex)
        fft_db = 20 * np.log10(fft_mag + 1e-10)
        fft_phase = np.angle(fft_complex)
        freqs = np.fft.rfftfreq(len(signal_to_analyze), 1 / sr_ana)

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("⏱️ מישור הזמן")
            fig_t, ax_t = plt.subplots(figsize=(6, 3.5))
            fig_t.patch.set_facecolor('#0e1117'); ax_t.set_facecolor('#0e1117')
            ax_t.plot(signal_to_analyze[:min(len(signal_to_analyze), 2000)], color='#ff007f', linewidth=1.2)
            ax_t.tick_params(colors='white'); ax_t.grid(True, linestyle='--', alpha=0.3)
            for s in ax_t.spines.values(): s.set_color('#30363d')
            st.pyplot(fig_t)

        with g2:
            if not show_phase:
                st.subheader("📊 מישור התדר (Magnitude Spectrum)")
                fig_f, ax_f = plt.subplots(figsize=(6, 3.5))
                fig_f.patch.set_facecolor('#0e1117'); ax_f.set_facecolor('#0e1117')
                ax_f.plot(freqs, fft_db, color='#00ffcc', linewidth=1.2)
                ax_f.set_xlim(0, sr_ana / 2)
                
                peak_idx = np.argmax(fft_mag)
                peak_freq = freqs[peak_idx]
                peak_db = fft_db[peak_idx]
                ax_f.annotate(f'Peak: {peak_freq:.1f}Hz', xy=(peak_freq, peak_db), xytext=(peak_freq, peak_db + 10),
                              arrowprops=dict(facecolor='#ffaa00', shrink=0.05, width=1, headwidth=5),
                              color='white', fontsize=9)

                ax_f.tick_params(colors='white'); ax_f.grid(True, linestyle='--', alpha=0.3)
                for s in ax_f.spines.values(): s.set_color('#30363d')
                st.pyplot(fig_f)
            else:
                st.subheader("🔄 תגובת פאזה (Phase Response)")
                fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
                fig_p.patch.set_facecolor('#0e1117'); ax_p.set_facecolor('#0e1117')
                ax_p.plot(freqs, fft_phase, color='#9b59b6', linewidth=1.2)
                ax_p.set_xlim(0, sr_ana / 2)
                ax_p.set_ylabel("פאזה (רדיאנים)", color='white')
                ax_p.tick_params(colors='white'); ax_p.grid(True, linestyle='--', alpha=0.3)
                for s in ax_p.spines.values(): s.set_color('#30363d')
                st.pyplot(fig_p)

        # --- מדדים וייצוא ---
        st.markdown("---")
        st.subheader("📈 מדדי DSP וייצוא נתונים")
        rms_val = np.sqrt(np.mean(signal_to_analyze**2))
        crest_factor = np.max(np.abs(signal_to_analyze)) / (rms_val + 1e-10)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("תדירות נייקוויסט", f"{sr_ana / 2:,.0f} Hz")
        m2.metric("התדר הדומיננטי (Peak)", f"{freqs[np.argmax(fft_mag)]:,.1f} Hz")
        m3.metric("הספק ממוצע (RMS)", f"{rms_val:.4f}")
        m4.metric("גורם הפסגה (Crest Factor)", f"{crest_factor:.2f}")

        csv_data = np.column_stack((t_ana, signal_to_analyze))
        csv_buffer = io.BytesIO()
        np.savetxt(csv_buffer, csv_data, delimiter=",", header="Time,Amplitude", comments="")
        st.download_button(
            label="📥 הורד את נתוני האות כקובץ CSV",
            data=csv_buffer.getvalue(),
            file_name="dsp_signal_data.csv",
            mime="text/csv"
        )
    else:
        st.info("בחר מקור אות או הפעל נתונים כדי לצפות בגרפים.")


# ==========================================
# כרטיסייה 2: מחולל אותות
# ==========================================
with tab_generator:
    st.header("מחולל אותות מתקדם (Signal Generator)")
    gen_type = st.selectbox("בחר צורת גל ליצירה:", ["גל סינוס טהור (Sine)", "גל מרובע (Square)", "גל משולש (Triangle)", "רעש לבן (White Noise)"], key="g_type")
    
    cg1, cg2 = st.columns(2)
    with cg1:
        gen_freq = st.slider("תדר האות (Hz)", 50, 5000, 440, key="g_f")
        gen_dur = st.slider("משך השמעה (שניות)", 0.5, 5.0, 2.0, 0.5, key="g_d")
    with cg2:
        gen_sr = st.selectbox("קצב דגימה", [8000, 16000, 32000, 44100], index=3, key="g_sr")
        gen_amp = st.slider("עוצמה", 0.1, 1.0, 0.5, key="g_amp")

    t_gen = np.linspace(0, gen_dur, int(gen_sr * gen_dur), endpoint=False)
    if "סינוס" in gen_type: gen_sig = gen_amp * np.sin(2 * np.pi * gen_freq * t_gen)
    elif "מרובע" in gen_type: gen_sig = gen_amp * np.sign(np.sin(2 * np.pi * gen_freq * t_gen))
    elif "משולש" in gen_type: gen_sig = gen_amp * signal_lib.sawtooth(2 * np.pi * gen_freq * t_gen, width=0.5)
    else: gen_sig = gen_amp * np.random.normal(0, 1, len(t_gen))

    fig_gen, ax_gen = plt.subplots(figsize=(10, 3))
    fig_gen.patch.set_facecolor('#0e1117'); ax_gen.set_facecolor('#0e1117')
    ax_gen.plot(t_gen[:1000], gen_sig[:1000], color='#ffaa00', linewidth=1.2)
    ax_gen.tick_params(colors='white'); ax_gen.grid(True, linestyle='--', alpha=0.3)
    for s in ax_gen.spines.values(): s.set_color('#30363d')
    st.pyplot(fig_gen)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🔊 השמע אות"): sd.play(gen_sig, gen_sr)
    with b2:
        scaled = np.int16(gen_sig / np.max(np.abs(gen_sig)) * 32767) if np.max(np.abs(gen_sig)) > 0 else gen_sig
        buf = io.BytesIO()
        wavfile.write(buf, gen_sr, scaled)
        st.download_button("💾 הורד WAV", buf.getvalue(), f"gen_{gen_freq}Hz.wav", "audio/wav")


# ==========================================
# כרטיסייה 3: תצוגת מפל מים (Waterfall / Spectrogram)
# ==========================================
with tab_waterfall:
    st.header("תצוגת ספקטרוגרמה / מפל מים (Waterfall Spectrogram)")
    st.write("מציג את התפתחות התדרים לאורך זמן בצבעים (חם = עוצמה גבוהה, קר = עוצמה נמוכה).")

    if signal_ana is not None and len(signal_ana) > 0:
        fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
        fig_wf.patch.set_facecolor('#0e1117'); ax_wf.set_facecolor('#0e1117')
        
        Pxx, freqs_w, bins, im = ax_wf.specgram(signal_ana, NFFT=1024, Fs=sr_ana, noverlap=512, cmap='inferno')
        
        ax_wf.set_xlabel("זמן (שניות)", color='white')
        ax_wf.set_ylabel("תדר (Hz)", color='white')
        cbar = fig_wf.colorbar(im, ax=ax_wf)
        cbar.set_label('עוצמה (dB)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        
        ax_wf.tick_params(colors='white')
        for s in ax_wf.spines.values(): s.set_color('#30363d')
        st.pyplot(fig_wf)
    else:
        st.info("אנא טען או בחר אות בכרטיסייה הראשונה כדי לראות את מפל המים.")
