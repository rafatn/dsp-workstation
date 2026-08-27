import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as signal_lib
import sympy as sp
import sqlite3
import pandas as pd
import io
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier

# --- הגדרת תצורת העמוד ---
st.set_page_config(
    page_title="Advanced DSP Workstation Ultimate Pro",
    page_icon="🎛️",
    layout="wide"
)

# --- עיצוב CSS מתקדם לשיפור נראות ואסתטיקה ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #161b22; }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-title { color: #9ca3af; font-size: 14px; margin-bottom: 5px; }
    .metric-value { color: #00ffcc; font-size: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🎛️ Advanced DSP Workstation Ultimate Pro")
st.markdown("תחנת עבודה הנדסית משולבת: **ניתוח ספקטרום, עיצוב UI מתקדם, חיבור למסדי נתונים וזיהוי אותות באמצעות למידת מכונה (ML)**.")

# --- יצירת כרטיסיות ראשיות ---
tab_analyzer, tab_generator, tab_transforms, tab_ml, tab_waterfall = st.tabs([
    "📊 1. ספקטרום אנלייזר + DB", 
    "📈 2. מחולל אותות", 
    "🧮 3. טרנספורם פורייה, לפלס וטורי טיילור",
    "🤖 4. זיהוי אותות וחריגות (ML)",
    "🌊 5. תצוגת מפל מים"
])

# ==========================================
# כרטיסייה 1: ספקטרום אנלייזר + מסד נתונים (SQL/CSV)
# ==========================================
with tab_analyzer:
    st.header("מנתח ספקטרום מתקדם עם תמיכה במסדי נתונים")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        analyzer_source = st.selectbox(
            "בחר מקור אות לניתוח:",
            ["גלים סינתטיים מורכבים", "גלים לא-הרמוניים (ריבועי/שן-מסור)", "רעש סטטיסטי", "העלאת קובץ CSV/WAV", "שליפה ממסד נתונים (SQLite)"],
            key="ana_src"
        )
        window_type = st.selectbox("סוג חלון ל-FFT:", ["Hann", "Hamming", "Blackman", "Rectangular (ללא)"], key="win_t")

    with col_c2:
        sr_ana = st.slider("קצב דגימה (Hz)", 8000, 48000, 44100, 1000, key="sr_a")
        duration_ana = st.slider("משך חלון (שניות)", 0.1, 3.0, 1.0, 0.1, key="dur_a")
        noise_boost = st.slider("הזרקת רעש לבן (SNR)", 0.0, 1.0, 0.0, 0.05, key="noise_b")

    with col_c3:
        st.subheader("⚙️ עיוותים ומסננים")
        distortion_drive = st.slider("עיוות לא-ליניארי / Clipping (Drive)", 1.0, 10.0, 1.0, 0.5)
        apply_filter = st.checkbox("הפעל מסנן (Filter)")
        filter_type = st.selectbox("סוג מסנן:", ["Low-Pass (מעביר-נמוכים)", "High-Pass (מעביר-גבוהים)", "Band-Pass (מעביר-רצועה)"])
        
        if "Band-Pass" in filter_type:
            c_low = st.slider("תדר חיתוך תחתון (Hz)", 100, 5000, 500, 50)
            c_high = st.slider("תדר חיתוך עליון (Hz)", 1000, 20000, 4000, 100)
        else:
            cutoff_freq = st.slider("תדר חיתוך (Cutoff Hz)", 100, 10000, 1500, 100)

    num_samples_ana = int(sr_ana * duration_ana)
    t_ana = np.linspace(0, duration_ana, num_samples_ana, endpoint=False)
    signal_ana = None
    info_ana = ""

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
            info_ana = f"גל ריבועי ב-{base_freq}Hz"
        else:
            signal_ana = 2 * (t_ana * base_freq - np.floor(0.5 + t_ana * base_freq))
            info_ana = f"גל שן-מסור ב-{base_freq}Hz"

    elif analyzer_source == "רעש סטטיסטי":
        signal_ana = np.random.normal(0, 1, num_samples_ana)
        info_ana = "רעש לבן אקראי"

    elif analyzer_source == "העלאת קובץ CSV/WAV":
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

    elif analyzer_source == "שליפה ממסד נתונים (SQLite)":
        st.info("מייצר בסיס נתונים מקומי לדוגמה ושולף ממנו רצף עיתי...")
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, amplitude REAL)")
        db_sig = np.sin(2 * np.pi * 300 * t_ana) + 0.2 * np.random.normal(0, 1, len(t_ana))
        cursor.executemany("INSERT INTO signals (amplitude) VALUES (?)", [(val,) for val in db_sig])
        conn.commit()
        
        df_db = pd.read_sql("SELECT amplitude FROM signals", conn)
        signal_ana = df_db['amplitude'].values
        conn.close()
        info_ana = "נתונים שנשלפו מטבלת SQLite בזיכרון"

    if signal_ana is not None and len(signal_ana) > 0:
        if noise_boost > 0:
            signal_ana = signal_ana + noise_boost * np.random.normal(0, 1, len(signal_ana))
            info_ana += f" + רעש ({noise_boost})"

        if distortion_drive > 1.0:
            signal_ana = np.clip(signal_ana * distortion_drive, -1.0, 1.0)
            info_ana += f" | דיסטורשן (Drive: {distortion_drive}x)"

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
        
        if "Hann" in window_type: window = np.hanning(len(signal_to_analyze))
        elif "Hamming" in window_type: window = np.hamming(len(signal_to_analyze))
        elif "Blackman" in window_type: window = np.blackman(len(signal_to_analyze))
        else: window = np.ones(len(signal_to_analyze))

        windowed = signal_to_analyze * window
        fft_complex = np.fft.rfft(windowed)
        fft_mag = np.abs(fft_complex)
        fft_db = 20 * np.log10(fft_mag + 1e-10)
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

        st.markdown("---")
        st.subheader("📈 מדדי DSP הנדסיים")
        
        rms_val = np.sqrt(np.mean(signal_to_analyze**2))
        crest_factor = np.max(np.abs(signal_to_analyze)) / (rms_val + 1e-10)
        dt = 1.0 / sr_ana
        signal_energy = np.sum(signal_to_analyze**2) * dt
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">תדירות נייקוויסט</div><div class="metric-value">{sr_ana / 2:,.0f} Hz</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">תדר דומיננטי</div><div class="metric-value">{freqs[np.argmax(fft_mag)]:,.1f} Hz</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">אנרגיית האות (E)</div><div class="metric-value">{signal_energy:.4f}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">הספק ממוצע (RMS)</div><div class="metric-value">{rms_val:.4f}</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-title">גורם הפסגה</div><div class="metric-value">{crest_factor:.2f}</div></div>', unsafe_allow_html=True)
    else:
        st.info("בחר מקור אות או טען נתונים כדי לצפות בגרפים.")


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

    scaled = np.int16(gen_sig / np.max(np.abs(gen_sig)) * 32767) if np.max(np.abs(gen_sig)) > 0 else gen_sig
    buf = io.BytesIO()
    wavfile.write(buf, gen_sr, scaled)
    st.download_button("💾 הורד קובץ שמע WAV להשמעה", buf.getvalue(), f"gen_{gen_freq}Hz.wav", "audio/wav")


# ==========================================
# כרטיסייה 3: טרנספורם פורייה, לפלס וטורי טיילור
# ==========================================
with tab_transforms:
    st.header("🧮 מחשבון טרנספורם פורייה, לפלס וקירוב טיילור/מקלורין")
    st.write("הקלד פונקציה בזמן $t$ (למשל: `exp(-2*t)*sin(3*t)`, `cos(t)`) וקבל את ההעתקים והקירובים הפולינומיים שלה.")

    t_sym = sp.Symbol('t', real=True, positive=True)
    omega = sp.Symbol('omega', real=True)
    s_sym = sp.Symbol('s')

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        example_func = st.selectbox(
            "בחר דוגמה מוכנה מראש:",
            ["הקלד פונקציה ידנית", "exp(-2*t)*sin(5*t)", "exp(-3*t)", "t * exp(-t)", "cos(4*t)", "sin(t)"]
        )
    with col_t2:
        taylor_order = st.slider("סדר פולינום לטור טיילור/מקלורין ($n$):", 1, 10, 4)
        taylor_point = st.number_input("נקודת פיתוח לטור טיילור ($t_0$):", value=0.0, step=0.5)

    if example_func == "הקלד פונקציה ידנית":
        func_input_str = st.text_input("הקלד פונקציה לפי $t$:", "exp(-2*t)*sin(3*t)")
    else:
        func_input_str = example_func
        st.info(f"נבחרה הפונקציה: `{func_input_str}`")

    if st.button("חשב טרנספורמים וטורים"):
        try:
            local_dict = {'t': t_sym, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'log': sp.log, 'Heaviside': sp.Heaviside}
            f_t = sp.sympify(func_input_str, locals=local_dict)

            st.markdown("### 📝 הפונקציה המקורית בזמן: $f(t)$")
            st.latex(f"f(t) = {sp.latex(f_t)}")

            # טרנספורם לפלס
            with st.spinner("מחשב טרנספורם לפלס..."):
                laplace_res = sp.laplace_transform(f_t, t_sym, s_sym, noconds=True)
            st.markdown("### ⚡ טרנספורם לפלס: $F(s) = \\mathcal{L}\\{f(t)\\}$")
            st.latex(f"F(s) = {sp.latex(laplace_res)}")

            # טרנספורם פורייה
            with st.spinner("מחשב טרנספורם פורייה..."):
                fourier_res = sp.fourier_transform(f_t, t_sym, omega)
            st.markdown("### 📊 טרנספורם פורייה רציף: $\\mathcal{F}\\{\\omega\\}$")
            st.latex(f"\\hat{f}(\\omega) = {sp.latex(fourier_res)}")

            # טור טיילור / מקלורין
            with st.spinner("מחשב טור טיילור/מקלורין..."):
                taylor_res = f_t.series(t_sym, taylor_point, taylor_order + 1).removeO()
            
            series_name = "מקלורין (סביב 0)" if taylor_point == 0 else f"טיילור (סביב t={taylor_point})"
            st.markdown(f"### 📈 פיתוח טור **{series_name}** (עד סדר {taylor_order}):")
            st.latex(f"f(t) \\approx {sp.latex(taylor_res)}")

        except Exception as e:
            st.error(f"שגיאה בניתוח הפונקציה: {e}.")


# ==========================================
# כרטיסייה 4: למידת מכונה לזיהוי אותות וחריגות (ML)
# ==========================================
with tab_ml:
    st.header("🤖 זיהוי סוגי אותות ואיתור חריגות באמצעות למידת מכונה (ML)")
    st.write("המודל מנתח את מאפייני הספקטרום של האות הנוכחי (מכרטיסייה 1) ומזהה את סוגו או האם יש בו חריגות סטטיסטיות (Anomalies).")

    if signal_ana is not None and len(signal_ana) > 0:
        col_ml1, col_ml2 = st.columns(2)

        with col_ml1:
            st.subheader("🔍 סיווג חכם של סוג האות")
            np.random.seed(42)
            X_train = np.array([
                [0.707, 1.5], [0.707, 1.4],
                [1.000, 5.2], [0.995, 5.0],
                [0.990, 8.5], [1.010, 9.0]
            ])
            y_train = ["גל סינוס", "גל סינוס", "גל ריבועי", "גל ריבועי", "רעש סטטיסטי", "רעש סטטיסטי"]
            
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X_train, y_train)

            curr_rms = np.sqrt(np.mean(signal_to_analyze**2))
            curr_crest = np.max(np.abs(signal_to_analyze)) / (curr_rms + 1e-10)
            prediction = clf.predict([[curr_rms, curr_crest]])[0]

            st.success(f"המודל סיווג את האות כדפוס: **{prediction}**")
            st.write(f"נתונים שנלמדו לצורך הסיווג: RMS = `{curr_rms:.3f}`, Crest Factor = `{curr_crest:.3f}`")

        with col_ml2:
            st.subheader("🚨 זיהוי חריגות (Anomaly Detection)")
            iso = IsolationForest(contamination=0.1, random_state=42)
            reshaped_sig = signal_to_analyze.reshape(-1, 1)
            iso.fit(reshaped_sig)
            anomalies = iso.predict(reshaped_sig)
            anomaly_count = np.sum(anomalies == -1)

            if anomaly_count > (len(signal_to_analyze) * 0.05):
                st.warning(f"⚠️ אזהרה: זוהו חריגות או עיוותים חריגים באות! (נמצאו {anomaly_count} נקודות חריגות)")
            else:
                st.info(f"✅ האות יציב ותקין מבחינה סטטיסטית (חריגות מינוריות: {anomaly_count} נקודות).")
                
            fig_ml, ax_ml = plt.subplots(figsize=(6, 2.5))
            fig_ml.patch.set_facecolor('#0e1117'); ax_ml.set_facecolor('#0e1117')
            ax_ml.plot(signal_to_analyze[:500], color='#00ffcc', label='Signal', linewidth=1)
            ax_ml.tick_params(colors='white'); ax_ml.grid(True, linestyle='--', alpha=0.3)
            for s in ax_ml.spines.values(): s.set_color('#30363d')
            st.pyplot(fig_ml)
    else:
        st.info("אנא טען או בחר אות בכרטיסייה הראשונה כדי להפעיל את מודלי ה-ML.")


# ==========================================
# כרטיסייה 5: תצוגת מפל מים (Waterfall)
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
