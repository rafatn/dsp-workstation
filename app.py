import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io.wavfile as wavfile
import scipy.signal as signal_lib
import sympy as sp
import sqlite3
import pandas as pd
import io
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# --- הגדרת תצורת העמוד ---
st.set_page_config(
    page_title="Advanced DSP Workstation Ultimate Pro v2",
    page_icon="🎛️",
    layout="wide"
)

# --- עיצוב CSS מתקדם ---
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

st.title("🎛️ Advanced DSP Workstation Ultimate Pro v2")
st.markdown("תחנת עבודה הנדסית מתקדמת: **השוואת אותות, מסנן LMS אדפטיבי, צפסטרום, דוחות PDF, ניתוח PSD, ו-ML**.")

# --- כרטיסיות ראשיות ---
tab_analyzer, tab_compare, tab_adaptive, tab_cepstrum, tab_generator, tab_transforms, tab_ml, tab_waterfall = st.tabs([
    "📊 1. ספקטרום אנלייזר + PSD + ייצוא", 
    "⚖️ 2. השוואת אותות כפולה",
    "🔄 3. סינון אדפטיבי (LMS)",
    "📡 4. ניתוח צפסטרום (Cepstrum)",
    "📈 5. מחולל אותות ואפקטים", 
    "🧮 6. פורייה, לפלס וטורי טיילור",
    "🤖 7. זיהוי אותות וחריגות (ML)",
    "🌊 8. מפל מים (Waterfall)"
])

# ==========================================
# כרטיסייה 1: ספקטרום אנלייזר + PSD + ייצוא
# ==========================================
with tab_analyzer:
    st.header("מנתח ספקטרום ראשי, PSD ודוחות")
    
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
        st.subheader("⚙️ עיוותים וסננונים מתקדמים")
        distortion_drive = st.slider("עיוות לא-ליניארי / Clipping (Drive)", 1.0, 10.0, 1.0, 0.5)
        apply_filter = st.checkbox("הפעל מסנן מתקדם (Filter)")
        filter_family = st.selectbox("משפחת מסנן:", ["Butterworth", "Chebyshev I", "Elliptic"])
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
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE signals (id INTEGER PRIMARY KEY, amplitude REAL)")
        db_sig = np.sin(2 * np.pi * 300 * t_ana) + 0.2 * np.random.normal(0, 1, len(t_ana))
        cursor.executemany("INSERT INTO signals (amplitude) VALUES (?)", [(val,) for val in db_sig])
        conn.commit()
        df_db = pd.read_sql("SELECT amplitude FROM signals", conn)
        signal_ana = df_db['amplitude'].values
        conn.close()
        info_ana = "נתונים שנשלפו מטבלת SQLite"

    if signal_ana is not None and len(signal_ana) > 0:
        if noise_boost > 0:
            signal_ana = signal_ana + noise_boost * np.random.normal(0, 1, len(signal_ana))
            info_ana += f" + רעש ({noise_boost})"

        if distortion_drive > 1.0:
            signal_ana = np.clip(signal_ana * distortion_drive, -1.0, 1.0)
            info_ana += f" | דיסטורשן ({distortion_drive}x)"

        if apply_filter:
            nyq = sr_ana / 2
            btype = 'band' if "Band-Pass" in filter_type else ('low' if 'Low' in filter_type else 'high')
            if "Band-Pass" in filter_type:
                low_norm = max(0.01, min(0.98, c_low / nyq))
                high_norm = max(low_norm + 0.01, min(0.99, c_high / nyq))
                Wn = [low_norm, high_norm]
            else:
                Wn = max(0.01, min(0.99, cutoff_freq / nyq))

            if filter_family == "Butterworth": b, a = signal_lib.butter(4, Wn, btype=btype)
            elif filter_family == "Chebyshev I": b, a = signal_lib.cheby1(4, 1.0, Wn, btype=btype)
            else: b, a = signal_lib.ellip(4, 1.0, 40, Wn, btype=btype)
            
            signal_to_analyze = signal_lib.lfilter(b, a, signal_ana)
            info_ana += f" | מסונן ({filter_family})"
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
        freqs_psd, psd_vals = signal_lib.welch(signal_to_analyze, fs=sr_ana, nperseg=min(1024, len(signal_to_analyze)))

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("⏱️ מישור הזמן")
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(y=signal_to_analyze[:min(len(signal_to_analyze), 2000)], mode='lines', line=dict(color='#ff007f', width=1.2)))
            fig_t.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig_t, use_container_width=True)

        with g2:
            st.subheader("📊 ספקטרום תדרים (Magnitude)")
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=freqs, y=fft_db, mode='lines', line=dict(color='#00ffcc', width=1.2)))
            fig_f.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), xaxis=dict(range=[0, sr_ana/2]), margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig_f, use_container_width=True)

        st.subheader("📈 צפיפות הספק (Welch PSD)")
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs_psd, y=10*np.log10(psd_vals + 1e-10), mode='lines', line=dict(color='#ffaa00', width=1.5)))
        fig_psd.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), xaxis_title="תדר (Hz)", yaxis_title="dB/Hz", margin=dict(l=20, r=20, t=20, b=20), height=260)
        st.plotly_chart(fig_psd, use_container_width=True)

        rms_val = np.sqrt(np.mean(signal_to_analyze**2))
        crest_factor = np.max(np.abs(signal_to_analyze)) / (rms_val + 1e-10)
        dt = 1.0 / sr_ana
        signal_energy = np.sum(signal_to_analyze**2) * dt
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-title">נייקוויסט</div><div class="metric-value">{sr_ana / 2:,.0f} Hz</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-title">תדר דומיננטי</div><div class="metric-value">{freqs[np.argmax(fft_mag)]:,.1f} Hz</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-title">אנרגיה (E)</div><div class="metric-value">{signal_energy:.4f}</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="metric-card"><div class="metric-title">RMS</div><div class="metric-value">{rms_val:.4f}</div></div>', unsafe_allow_html=True)
        with m5: st.markdown(f'<div class="metric-card"><div class="metric-title">גורם פסגה</div><div class="metric-value">{crest_factor:.2f}</div></div>', unsafe_allow_html=True)

        # ייצוא ל-Excel ול-PDF
        st.markdown("<br>", unsafe_allow_html=True)
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            df_export = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude_dB': fft_db})
            buffer_excel = io.BytesIO()
            with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, sheet_name='FFT_Data', index=False)
            st.download_button("📥 הורד נתוני ספקטרום (Excel)", buffer_excel.getvalue(), "dsp_spectrum.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with col_ex2:
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "Advanced DSP Workstation - Engineering Report")
            c.drawString(100, 730, f"Signal Info: {info_ana}")
            c.drawString(100, 700, f"Sampling Rate: {sr_ana} Hz")
            c.drawString(100, 680, f"RMS Power: {rms_val:.4f}")
            c.drawString(100, 660, f"Crest Factor: {crest_factor:.2f}")
            c.drawString(100, 640, f"Dominant Frequency: {freqs[np.argmax(fft_mag)]:,.1f} Hz")
            c.save()
            st.download_button("📄 הורד דוח הנדסי (PDF)", pdf_buffer.getvalue(), "dsp_report.pdf", "application/pdf")
    else:
        st.info("אנא בחר מקור אות בכרטיסייה זו.")

# ==========================================
# כרטיסייה 2: השוואת אותות כפולה
# ==========================================
with tab_compare:
    st.header("⚖️ השוואת שני אותות במקביל (Dual Signal Comparison)")
    st.write("השווה בין אות מקור לאות מושווה או מסונן באותו ציר זמן ותדר.")
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.subheader("אות ראשון (Signal A)")
        f_a = st.slider("תדר אות A (Hz)", 50, 2000, 440, key="fa")
        sig_a = np.sin(2 * np.pi * f_a * np.linspace(0, 0.5, 22050))
    with col_comp2:
        st.subheader("אות שני (Signal B)")
        f_b = st.slider("תדר אות B (Hz)", 50, 2000, 880, key="fb")
        sig_b = np.sin(2 * np.pi * f_b * np.linspace(0, 0.5, 22050)) + 0.3 * np.random.normal(0, 1, 22050)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(y=sig_a[:1000], name="אות A", line=dict(color='#00ffcc', width=1.2)))
    fig_comp.add_trace(go.Scatter(y=sig_b[:1000], name="אות B", line=dict(color='#ff007f', width=1.2)))
    fig_comp.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

# ==========================================
# כרטיסייה 3: סינון אדפטיבי (LMS Filter)
# ==========================================
with tab_adaptive:
    st.header("🔄 סינון אדפטיבי מבוסס Least Mean Squares (LMS)")
    st.write("סינון רעשים דינמי מאות רועש בזמן אמת.")
    
    n_pts = 1000
    t_lms = np.linspace(0, 1, n_pts)
    desired = np.sin(2 * np.pi * 5 * t_lms)
    noise_lms = 0.5 * np.random.normal(0, 1, n_pts)
    input_sig = desired + noise_lms

    mu = st.slider("קצב למידה (Learning Rate - Mu)", 0.001, 0.1, 0.01, 0.005)
    filter_order = st.slider("מספר מקדמי המסנן (Taps)", 2, 32, 8)
    
    w = np.zeros(filter_order)
    y_out = np.zeros(n_pts)
    e_out = np.zeros(n_pts)
    
    for n in range(filter_order, n_pts):
        x_vec = input_sig[n-filter_order:n][::-1]
        y_out[n] = np.dot(w, x_vec)
        e_out[n] = desired[n] - y_out[n]
        w += 2 * mu * e_out[n] * x_vec

    fig_lms = go.Figure()
    fig_lms.add_trace(go.Scatter(y=input_sig, name="אות נכנס רועש", line=dict(color='#ff4444', width=1)))
    fig_lms.add_trace(go.Scatter(y=e_out, name="שגיאה / אות מסונן", line=dict(color='#00ffcc', width=1.5)))
    fig_lms.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_lms, use_container_width=True)

# ==========================================
# כרטיסייה 4: ניתוח צפסטרום (Cepstrum)
# ==========================================
with tab_cepstrum:
    st.header("📡 ניתוח צפסטרום (Cepstrum Analysis)")
    st.write("משמש לזיהוי תדרים בסיסיים נסתרים (Pitch) ואיתור הדים (Echoes).")
    
    if signal_ana is not None and len(signal_ana) > 0:
        spectrum = np.abs(np.fft.fft(signal_ana))
        log_spec = np.log(spectrum + 1e-10)
        cepstrum = np.abs(np.fft.ifft(log_spec))
        quefrency = np.linspace(0, len(signal_ana)/sr_ana, len(cepstrum))

        fig_cep = go.Figure()
        fig_cep.add_trace(go.Scatter(x=quefrency[:len(quefrency)//2], y=cepstrum[:len(cepstrum)//2], mode='lines', line=dict(color='#aa00ff', width=1.5)))
        fig_cep.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), xaxis_title="Quefrency (שניות)", yaxis_title="אמפליטודה", height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cep, use_container_width=True)
    else:
        st.info("אנא טען או בחר אות בכרטיסייה הראשונה.")

# ==========================================
# כרטיסייה 5: מחולל אותות ואפקטים
# ==========================================
with tab_generator:
    st.header("מחולל אותות ואפקטים קוליים")
    gen_type = st.selectbox("צורת גל:", ["גל סינוס", "גל מרובע", "רעש לבן"], key="g_type")
    gen_freq = st.slider("תדר (Hz)", 50, 5000, 440, key="g_f")
    gen_dur = st.slider("משך (שניות)", 0.5, 3.0, 1.0, key="g_d")
    
    t_gen = np.linspace(0, gen_dur, int(44100 * gen_dur), endpoint=False)
    gen_sig = np.sin(2 * np.pi * gen_freq * t_gen) if "סינוס" in gen_type else np.random.normal(0, 1, len(t_gen))
    
    fig_gen = go.Figure()
    fig_gen.add_trace(go.Scatter(y=gen_sig[:1000], line=dict(color='#ffaa00', width=1.2)))
    fig_gen.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), height=250, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_gen, use_container_width=True)

# ==========================================
# כרטיסייה 6: פורייה, לפלס וטורי טיילור
# ==========================================
with tab_transforms:
    st.header("🧮 מחשבון מתמטי (פורייה, לפלס וטורי טיילור)")
    
    transform_type = st.selectbox(
        "בחר כלי מתמטי:", 
        ["התמרת לפלס (Laplace)", "טור טיילור (Taylor Series)", "טור פוריאה (Fourier Series)"],
        key="trans_select_box"
    )
    
    t_sym = sp.Symbol('t', real=True, positive=True)
    x_sym = sp.Symbol('x', real=True)
    
    if "לפלס" in transform_type:
        func_input_str = st.text_input("הקלד פונקציה לפי $t$:", "exp(-2*t)*sin(3*t)", key="lap_input")
        if st.button("חשב לפלס", key="btn_laplace"):
            try:
                f_t = sp.sympify(func_input_str)
                laplace_res = sp.laplace_transform(f_t, t_sym, sp.Symbol('s'), noconds=True)
                st.latex(rf"f(t) = {sp.latex(f_t)}")
                st.latex(rf"\mathcal{{L}}\{{{sp.latex(f_t)}\}} = {sp.latex(laplace_res)}")
            except Exception as e:
                st.error(f"שגיאה בפענוח הפונקציה: {e}")

    elif "טיילור" in transform_type:
        func_input_str = st.text_input("הקלד פונקציה לפי $x$:", "exp(x)", key="taylor_input")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            taylor_point = st.number_input("נקודת פיתוח ($x_0$)", value=0.0, key="taylor_pt")
        with col_t2:
            taylor_order = st.slider("דרגה (Order)", 1, 10, 4, key="taylor_ord")
            
        if st.button("חשב טורי טיילור", key="btn_taylor"):
            try:
                f_x = sp.sympify(func_input_str)
                taylor_res = f_x.series(x_sym, taylor_point, taylor_order + 1).removeO()
                st.latex(rf"f(x) = {sp.latex(f_x)}")
                st.latex(rf"T_n(x) \approx {sp.latex(taylor_res)}")
            except Exception as e:
                st.error(f"שגיאה בפיתוח טיילור: {e}")

    elif "פוריאה" in transform_type:
        st.subheader("טור פוריאה (Fourier Series Expansion)")
        st.write("מציאת מקדמי טור פוריאה לפונקציה מחזורית בתוך התחום $[-\\pi, \\pi]$.")
        func_input_str = st.text_input("הקלד פונקציה מחזורית לפי $x$:", "x", key="fourier_input")
        fourier_n = st.slider("מספר הרמוניות ($N$)", 1, 10, 3, key="fourier_n_slider")
        
        if st.button("חשב טור פוריאה", key="btn_fourier"):
            try:
                f_x = sp.sympify(func_input_str)
                L = sp.pi
                
                a0 = (1 / (2 * L)) * sp.integrate(f_x, (x_sym, -L, L)).doit()
                fourier_sum = a0
                terms_display = [sp.latex(a0)]
                
                for n in range(1, fourier_n + 1):
                    an = (1 / L) * sp.integrate(f_x * sp.cos(n * x_sym), (x_sym, -L, L)).doit()
                    bn = (1 / L) * sp.integrate(f_x * sp.sin(n * x_sym), (x_sym, -L, L)).doit()
                    
                    if an != 0:
                        fourier_sum += an * sp.cos(n * x_sym)
                        terms_display.append(rf"{sp.latex(an)} \cos({n}x)")
                    if bn != 0:
                        fourier_sum += bn * sp.sin(n * x_sym)
                        terms_display.append(rf"{sp.latex(bn)} \sin({n}x)")
                
                st.latex(rf"f(x) = {sp.latex(f_x)}")
                st.latex(rf"S_N(x) \approx " + " + ".join(terms_display))
            except Exception as e:
                st.error(f"שגיאה בחישוב טור פוריאה: {e}")

# ==========================================
# כרטיסייה 7: למידת מכונה (ML)
# ==========================================
with tab_ml:
    st.header("🤖 זיהוי סוגי אותות וחריגות (ML)")
    if signal_ana is not None and len(signal_ana) > 0:
        iso = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso.fit_predict(signal_ana.reshape(-1, 1))
        st.info(f"זוהו {np.sum(anomalies == -1)} נקודות חריגות באות.")
    else:
        st.info("יש לטעון אות בכרטיסייה 1.")

# ==========================================
# כרטיסייה 8: מפל מים (Waterfall)
# ==========================================
with tab_waterfall:
    st.header("🌊 תצוגת מפל מים (Waterfall Spectrogram)")
    if signal_ana is not None and len(signal_ana) > 0:
        f_sgram, t_sgram, Sxx = signal_lib.spectrogram(signal_ana, fs=44100, nperseg=512)
        fig_wf = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx + 1e-10), x=t_sgram, y=f_sgram, colorscale='Inferno'))
        fig_wf.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'), height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.info("יש לטעון אות בכרטיסייה 1.")
