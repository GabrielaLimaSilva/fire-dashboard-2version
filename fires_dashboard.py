filename = 'Hear the Fire'
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pydub import AudioSegment
import os
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips, AudioClip
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from datetime import datetime
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

st.set_page_config(page_title=f'{filename}', layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .main .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 100% !important; }
        #MainMenu, footer, header { visibility: hidden; }
        body { background: #0a0a14; color: #f5f5f5; }
        
        .main-header { background: linear-gradient(135deg, #ff4444 0%, #ff8c00 50%, #ffd700 100%); padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(255, 68, 68, 0.4); }
        .main-header h1 { margin: 0; color: white; font-size: 28px; font-weight: 700; }
        .main-header p { margin: 0.3rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 13px; }
        
        .stat-card { background: linear-gradient(135deg, rgba(255, 68, 68, 0.12) 0%, rgba(255, 140, 0, 0.08) 100%); padding: 0.6rem 0.8rem; border-radius: 10px; border-left: 3px solid #ff4444; margin-bottom: 0.5rem; }
        .metric-label { font-size: 9px; color: #ff8c00; font-weight: 600; text-transform: uppercase; }
        .metric-value { font-size: 16px; color: #ffd700; font-weight: 700; }
        
        .video-container { background: #000; border-radius: 16px; overflow: visible; box-shadow: 0 12px 40px rgba(255, 68, 68, 0.5); border: 2px solid rgba(255, 140, 0, 0.3); height: calc(100vh - 220px); display: flex; align-items: center; justify-content: center; padding: 0.5rem; }
        
        .stButton>button { background: linear-gradient(135deg, #ff4444 0%, #ff8c00 100%) !important; color: white !important; border: none !important; padding: 0.6rem 1.2rem !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important; }
        
        .info-box { background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 140, 0, 0.1) 100%); border-left: 4px solid #ff4444; padding: 0.8rem; border-radius: 10px; margin: 0.8rem 0; font-size: 12px; }
        .info-box strong { color: #ffd700; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# ============= NOVA GERAÇÃO DE SOM - ENSEMBLE STRINGS =============

SAMPLE_RATE = 44100

def midi_to_hz(m):
    """Converte nota MIDI para frequência em Hz"""
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Notas D-F-A (acorde Dm) em 4 oitavas
NOTE_SET = []
base_notes = [50, 53, 57]  # D3, F3, A3
for octave in range(-1, 3):
    for n in base_notes:
        NOTE_SET.append(n + 12 * octave)
NOTE_SET = sorted([m for m in NOTE_SET if 36 <= 84])

def map_val_to_note(v, vmin, vmax):
    """Mapeia intensidade de fogo para nota MIDI"""
    if np.isnan(v) or vmax <= vmin:
        return NOTE_SET[len(NOTE_SET)//2]
    t = (v - vmin) / (vmax - vmin)
    idx = int(t * (len(NOTE_SET) - 1))
    return NOTE_SET[min(max(idx, 0), len(NOTE_SET)-1)]

def one_pole_lowpass(x, sr, cutoff_hz=2500.0):
    """Filtro passa-baixa simples"""
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sr)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y

def soft_env(w, sr, a_sec=0.25, r_sec=0.6):
    """Envelope de ataque e release suaves"""
    n = len(w)
    a = max(1, int(a_sec * sr))
    r = max(1, int(r_sec * sr))
    a = min(a, n//2)
    r = min(r, n - a)
    env = np.ones(n, dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r, dtype=np.float32)
    return w * env

def ensemble_strings(base_hz, n, sr):
    """Gera som de strings ensemble com vibrato e detuning"""
    t = np.arange(n, dtype=np.float32) / sr
    
    # Vibrato
    vib_rate = 5.0
    vib_depth = 0.006
    
    # Detuning para riqueza de ensemble
    cents = np.array([-7.0, 0.0, 7.0], dtype=np.float32)
    detune = 2.0 ** (cents / 1200.0)
    voices = np.zeros(n, dtype=np.float32)
    
    for d in detune:
        f_inst = base_hz * d * (1.0 + vib_depth * np.sin(2*np.pi*vib_rate*t))
        phase = 2*np.pi * np.cumsum(f_inst) / sr
        tri = (2.0/np.pi) * np.arcsin(np.sin(phase))
        voices += tri
    
    voices /= len(detune)
    
    # Adiciona ruído sutil
    noise = (np.random.uniform(-1.0, 1.0, n).astype(np.float32)) * 0.02
    y = voices + noise
    
    # Filtro passa-baixa
    y = one_pole_lowpass(y, sr, cutoff_hz=2300.0)
    
    # Envelope
    y = soft_env(y, sr, a_sec=0.18, r_sec=0.5)
    
    # Normaliza
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak)

def create_drone_layer(duration_sec, sr=44100):
    """Cria camada de drone com notas D2, D3, A3, D4"""
    t = np.arange(int(duration_sec * sr), dtype=np.float32) / sr
    
    d2 = 73.42   # D2
    d3 = 146.83  # D3
    a3 = 220.00  # A3
    d4 = 293.66  # D4
    
    drone = (
        0.15 * np.sin(2 * np.pi * d2 * t) +
        0.35 * np.sin(2 * np.pi * d3 * t) +
        0.35 * np.sin(2 * np.pi * a3 * t) +
        0.15 * np.sin(2 * np.pi * d4 * t)
    )
    
    # Fade in/out
    fade_len = int(2.5 * sr)
    if len(drone) > fade_len * 2:
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        drone[:fade_len] *= fade_in
        drone[-fade_len:] *= fade_out
    
    # Breathing suave
    mod = 0.75 + 0.25 * np.sin(2 * np.pi * 0.06 * t)
    drone *= mod
    
    # Volume baixo para não dominar
    drone *= 0.02
    
    return drone.astype(np.float32)

def compose_fire_ensemble(fires_per_day_df, total_duration_sec=14):
    """Compõe música usando ensemble strings para cada dia"""
    n_days = len(fires_per_day_df)
    
    # Calcula min/max para mapeamento
    vmin = fires_per_day_df['n_fires'].min()
    vmax = fires_per_day_df['n_fires'].max()
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    # Duração por dia
    seconds_per_day = total_duration_sec / n_days
    samples_per_day = int(SAMPLE_RATE * seconds_per_day)
    total_samples = int(SAMPLE_RATE * total_duration_sec)
    
    # Buffer principal
    buf = np.zeros(total_samples, dtype=np.float32)
    
    # Gera som para cada dia
    for day_idx, (day, n_fires) in enumerate(fires_per_day_df.values):
        s_idx = day_idx * samples_per_day
        e_idx = min((day_idx + 1) * samples_per_day, total_samples)
        n = e_idx - s_idx
        
        if n <= 0:
            continue
        
        # Mapeia número de fogos para nota MIDI
        midi_note = map_val_to_note(float(n_fires), vmin, vmax)
        freq = midi_to_hz(midi_note)
        
        # Gera string ensemble para este dia
        day_sound = ensemble_strings(freq, n, SAMPLE_RATE)
        
        # Adiciona ao buffer com volume ajustado
        buf[s_idx:e_idx] += 0.38 * day_sound
    
    # Normaliza antes de adicionar drone
    buf = np.tanh(buf / (np.max(np.abs(buf)) + 1e-9))
    
    # Adiciona drone
    drone = create_drone_layer(total_duration_sec, SAMPLE_RATE)
    buf += drone
    
    # Normalização final
    buf /= (np.max(np.abs(buf)) + 1e-9)
    buf = np.tanh(buf)
    
    # Converte para AudioSegment
    pcm16 = (buf * 32767).astype(np.int16).tobytes()
    audio_segment = AudioSegment(
        data=pcm16,
        sample_width=2,
        frame_rate=SAMPLE_RATE,
        channels=1
    ).apply_gain(-2)  # Master gain
    
    return audio_segment

# ============= FUNÇÕES AUXILIARES =============

def distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

st.markdown('<div class="main-header"><h1>🔥 Hear the Fire</h1><p>Transform fire data into an immersive audiovisual experience with ensemble strings</p></div>', unsafe_allow_html=True)

progress_placeholder = st.empty()
status_placeholder = st.empty()

st.sidebar.markdown("### ⚙️ Settings")

try:
    map_key = st.secrets["NASA_FIRMS_KEY"]
except:
    map_key = "a4abee84e580a96ff5ba9bd54cd11a8d"

col1, col2 = st.sidebar.columns(2)
with col1:
    latitude_center = st.number_input("Latitude", value=-19.0, step=0.1)
with col2:
    longitude_center = st.number_input("Longitude", value=-59.4, step=0.1)

radius_km = st.sidebar.slider("Radius (km)", 50, 1000, 150, 50)

col1, col2 = st.sidebar.columns(2)
with col1:
    data_date = st.date_input("Start date", value=datetime(2019, 8, 14)).strftime("%Y-%m-%d")
with col2:
    day_range = st.number_input("Days", value=10, min_value=1, max_value=30)

total_duration_sec = 1.2*day_range

os.makedirs("maps_png", exist_ok=True)

col_left, col_right = st.columns([1, 3], gap="medium")

with col_left:
    st.markdown('<div class="info-box"><strong>🎵 Ensemble Strings:</strong> Each day becomes a rich string chord. More fires = higher notes. <strong>Listen to the harmony of data.</strong></div>', unsafe_allow_html=True)
    
    if st.button("🔥 GENERATE", key="generate_btn"):
        st.session_state['generate_clicked'] = True
    
    if 'video_file' in st.session_state and os.path.exists(st.session_state.get('video_file', '')):
        st.markdown("#### 📊 Stats")
        if 'stats_data' in st.session_state:
            stats = st.session_state['stats_data']
            st.markdown(f'<div class="stats-grid"><div class="stat-card"><div class="metric-label">🔥 Total</div><div class="metric-value">{stats["total"]}</div></div><div class="stat-card"><div class="metric-label">📊 Days</div><div class="metric-value">{stats["days"]}</div></div><div class="stat-card"><div class="metric-label">📈 Avg</div><div class="metric-value">{stats["avg"]:.0f}</div></div><div class="stat-card"><div class="metric-label">⚡ Peak</div><div class="metric-value">{stats["peak"]}</div></div></div>', unsafe_allow_html=True)
        
        st.markdown("#### 💾 Download")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if 'mp3_file' in st.session_state and os.path.exists(st.session_state['mp3_file']):
                with open(st.session_state['mp3_file'], "rb") as f:
                    st.download_button("🎵 MP3", f.read(), st.session_state['mp3_file'], "audio/mpeg", use_container_width=True)
        with col_d2:
            with open(st.session_state['video_file'], "rb") as f:
                st.download_button("🎬 MP4", f.read(), st.session_state['video_file'], "video/mp4", use_container_width=True)

with col_right:
    if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
        st.markdown('<div class="video-container"><div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.8);"><h2 style="color: #ffd700;">⏳ Generating...</h2><p>Please wait.</p></div></div>', unsafe_allow_html=True)
    elif 'video_file' in st.session_state and os.path.exists(st.session_state.get('video_file', '')):
        st.markdown("### 🎬 Your Creation")
        st.video(st.session_state['video_file'])
    else:
        st.markdown('<div class="video-container"><div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.5);"><h2 style="color: #ffd700;">🎬 Your Video Will Appear Here</h2><p>Configure parameters and click GENERATE.</p></div></div>', unsafe_allow_html=True)

if 'generate_clicked' in st.session_state and st.session_state['generate_clicked']:
    progress_bar = progress_placeholder.progress(0)
    status_text = status_placeholder.empty()
    
    try:
        status_text.text("🔍 Fetching fire data from NASA...")
        progress_bar.progress(5)
        response = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_SP/world/{day_range}/{data_date}", timeout=30)
        progress_bar.progress(10)
        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.strip().str.lower()
        lat_col = next((c for c in df.columns if 'lat' in c), None)
        lon_col = next((c for c in df.columns if 'lon' in c), None)
        
        status_text.text("📊 Processing fire data...")
        progress_bar.progress(15)
        df['dist_km'] = distance_km(latitude_center, longitude_center, df[lat_col], df[lon_col])
        df_local = df[df['dist_km'] <= radius_km].copy()
        progress_bar.progress(20)
        
        if not df_local.empty:
            fires_per_day = df_local.groupby('acq_date').size().reset_index(name='n_fires')
            st.session_state['stats_data'] = {'total': len(df_local), 'days': len(fires_per_day), 'avg': fires_per_day['n_fires'].mean(), 'peak': fires_per_day['n_fires'].max()}
            
            status_text.text("🎵 Composing ensemble strings symphony...")
            progress_bar.progress(25)
            melody = compose_fire_ensemble(fires_per_day, total_duration_sec)
            progress_bar.progress(35)
            melody.export("fires_sound.mp3", format="mp3", bitrate="192k")
            st.session_state['mp3_file'] = "fires_sound.mp3"
            progress_bar.progress(40)
            
            lon_min = longitude_center - radius_km/100
            lon_max = longitude_center + radius_km/100
            lat_min = latitude_center - radius_km/100
            lat_max = latitude_center + radius_km/100
            images_files = []
            all_days = fires_per_day['acq_date'].tolist()
            n_days = len(fires_per_day)
            n_fade_frames = 5
            intro_frames = 15
            
            status_text.text("🎬 Creating intro animation...")
            for i in range(intro_frames):
                progress = (i + 1) / intro_frames
                progress_bar.progress(40 + int(10 * progress))
                fig = plt.figure(figsize=(16, 9), dpi=100)
                fig.patch.set_facecolor('black')
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
                ax_bar = fig.add_subplot(gs[1])
                fig.patch.set_facecolor('#000000')
                ax_map.set_facecolor('black')
                ax_bar.set_facecolor('black')
                ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                ax_map.add_feature(cfeature.LAND, facecolor='none', edgecolor='gray', linewidth=0.8)
                ax_map.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
                ax_map.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5)
                ax_map.set_xticks([])
                ax_map.set_yticks([])
                ax_map.plot(longitude_center, latitude_center, 'ro', markersize=15, transform=ccrs.PlateCarree(), alpha=0.8)
                current_radius_km = radius_km * progress
                lat_deg_radius = current_radius_km / 111
                lon_deg_radius = current_radius_km / (111 * np.cos(np.radians(latitude_center)))
                theta = np.linspace(0, 2*np.pi, 100)
                lat_circle = latitude_center + lat_deg_radius * np.sin(theta)
                lon_circle = longitude_center + lon_deg_radius * np.cos(theta)
                ax_map.plot(lon_circle, lat_circle, 'r-', linewidth=2, transform=ccrs.PlateCarree(), alpha=0.7)
                if progress > 0.7:
                    lat_end = latitude_center + lat_deg_radius * np.sin(np.pi/4)
                    lon_end = longitude_center + lon_deg_radius * np.cos(np.pi/4)
                    ax_map.plot([longitude_center, lon_end], [latitude_center, lat_end], 'y-', linewidth=3, transform=ccrs.PlateCarree(), alpha=0.8)
                    mid_lat = (latitude_center + lat_end)/2
                    mid_lon = (longitude_center + lon_end)/2
                    ax_map.text(mid_lon, mid_lat, f'{radius_km} km', color='white', fontsize=16, fontweight='bold', transform=ccrs.PlateCarree(), ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
                ax_bar.set_facecolor('black')
                ax_bar.set_xlim(0, 1)
                ax_bar.set_ylim(0, 1)
                ax_bar.set_xticks([])
                ax_bar.set_yticks([])
                for spine in ax_bar.spines.values():
                    spine.set_visible(False)
                for spine in ax_map.spines.values():
                    spine.set_visible(False)
                png_file = f"maps_png/intro_{i}.png"
                fig.savefig(png_file, facecolor='#000000', dpi=100, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
                img = Image.open(png_file).convert("RGB")
                img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                img.save(png_file, quality=85, optimize=True)
                images_files.append(png_file)
            
            status_text.text("🔥 Rendering fire visualizations...")
            total_fire_frames = n_days * n_fade_frames
            for i, (day, n_fires) in enumerate(fires_per_day.values):
                status_text.text(f"🔥 Rendering day {i+1}/{n_days}: {day} ({n_fires} fires)")
                df_day = df_local[df_local['acq_date'] == day]
                frp_norm = np.zeros(len(df_day))
                if 'frp' in df_day.columns and not df_day['frp'].isna().all():
                    frp_norm = (df_day['frp'] - df_day['frp'].min()) / (df_day['frp'].max() - df_day['frp'].min() + 1e-6)
                for k in range(n_fade_frames):
                    frame_progress = (i * n_fade_frames + k) / total_fire_frames
                    progress_bar.progress(50 + int(40 * frame_progress))
                    alpha = (k+1)/n_fade_frames
                    fig = plt.figure(figsize=(16, 9), dpi=100)
                    fig.patch.set_facecolor('black')
                    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
                    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
                    ax_bar = fig.add_subplot(gs[1])
                    fig.patch.set_facecolor('#000000')
                    ax_map.set_facecolor('black')
                    ax_bar.set_facecolor('black')
                    ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                    ax_map.add_feature(cfeature.LAND, facecolor='none', edgecolor='gray', linewidth=0.8)
                    ax_map.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
                    ax_map.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5)
                    ax_map.set_xticks([])
                    ax_map.set_yticks([])
                    
                    if len(df_day) > 0:
                        glow_sizes = 400 + 100 * np.sin(alpha * np.pi * 2)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#8B0000', s=glow_sizes, alpha=0.15 * alpha,
                                     transform=ccrs.PlateCarree())
                        
                        halo_sizes = 250 + 80 * np.sin(alpha * np.pi * 2)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#FF4500', s=halo_sizes, alpha=0.25 * alpha,
                                     transform=ccrs.PlateCarree())
                        
                        core_sizes = 150 + 60 * np.sin(alpha * np.pi * 2)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c='#FF8C00', s=core_sizes, alpha=0.6 * alpha,
                                     linewidths=0, transform=ccrs.PlateCarree())
                        
                        center_colors = plt.cm.YlOrRd(frp_norm * 0.7 + 0.3)
                        center_sizes = 80 + 50 * np.sin(alpha * np.pi * 3) * (1 + frp_norm)
                        ax_map.scatter(df_day[lon_col], df_day[lat_col], 
                                     c=center_colors, s=center_sizes, alpha=0.85 * alpha,
                                     edgecolors='#FFD700', linewidths=1,
                                     transform=ccrs.PlateCarree())
                        
                        high_intensity = df_day[df_day['frp'] > df_day['frp'].quantile(0.7)] if 'frp' in df_day.columns else df_day.head(int(len(df_day)*0.3))
                        if len(high_intensity) > 0:
                            white_sizes = 60 + 40 * np.sin(alpha * np.pi * 4)
                            ax_map.scatter(high_intensity[lon_col], high_intensity[lat_col], 
                                         c='white', s=white_sizes, alpha=0.9 * alpha,
                                         edgecolors='#FFFF00', linewidths=1.5,
                                         transform=ccrs.PlateCarree(), marker='*', zorder=10)
                        
                        if k % 2 == 0:
                            burst_indices = np.random.choice(len(df_day), size=min(3, len(df_day)), replace=False)
                            burst_points = df_day.iloc[burst_indices]
                            ax_map.scatter(burst_points[lon_col], burst_points[lat_col],
                                         c='#FF0000', s=500, alpha=0.2,
                                         transform=ccrs.PlateCarree())
                    
                    bar_heights = [fires_per_day.loc[fires_per_day['acq_date']==d,'n_fires'].values[0] if d<=day else 0 for d in all_days]
                    colors = ['orangered' if d<=day else 'gray' for d in all_days]
                    bars = ax_bar.bar(all_days, bar_heights, color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
                    for bar, height in zip(bars, bar_heights):
                        if height > 0:
                            bar.set_linewidth(1.5)
                            bar.set_edgecolor('#ffd700')
                    ax_bar.tick_params(colors='white', labelsize=12)
                    ax_bar.set_ylabel('Number of Fires', color='white', fontsize=14, fontweight='bold')
                    ax_bar.set_xlabel('Date', color='white', fontsize=14, fontweight='bold')
                    ax_bar.set_ylim(0, fires_per_day['n_fires'].max()*1.2)
                    ax_bar.grid(axis='y', alpha=0.2, linestyle='--', color='gray')
                    ax_bar.set_facecolor('#0a0a0a')
                    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right')
                    for spine in ax_bar.spines.values():
                        spine.set_color('#ff8c00')
                        spine.set_linewidth(1.5)
                    for spine in ax_map.spines.values():
                        spine.set_visible(False)
                    ax_map.tick_params(left=False, right=False, top=False, bottom=False)
                    png_file = f"maps_png/map_{i}_{k}.png"
                    fig.savefig(png_file, facecolor='#000000', dpi=100, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)
                    img = Image.open(png_file).convert("RGB")
                    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    img.save(png_file, quality=85, optimize=True)
                    images_files.append(png_file)
            
            status_text.text("🎬 Assembling video...")
            progress_bar.progress(90)
            
            intro_duration = 4.0
            fires_duration = total_duration_sec
            intro_frame_duration = intro_duration / intro_frames
            fires_frame_count = len(images_files) - intro_frames
            fires_frame_duration = fires_duration / fires_frame_count if fires_frame_count > 0 else 0.1
            frame_durations = [intro_frame_duration] * intro_frames + [fires_frame_duration] * fires_frame_count
            
            clip = ImageSequenceClip(images_files, durations=frame_durations)
            clip = clip.on_color(size=(1280, 720), color=(0,0,0))
            audio_clip = AudioFileClip("fires_sound.mp3")
            
            def make_frame(t):
                return [0, 0]
            
            silent_audio = AudioClip(make_frame, duration=intro_duration, fps=44100)
            full_audio = concatenate_audioclips([silent_audio, audio_clip])
            clip = clip.set_audio(full_audio)
            clip.fps = 24
            
            status_text.text("💾 Exporting final video...")
            progress_bar.progress(95)
            clip.write_videofile("fires_video.mp4", codec="libx264", audio_codec="aac", verbose=False, logger=None)
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            st.session_state['video_file'] = "fires_video.mp4"
            st.session_state['generate_clicked'] = False
            progress_placeholder.empty()
            status_placeholder.empty()
            st.rerun()
        else:
            progress_placeholder.empty()
            status_placeholder.empty()
            st.error("⚠️ No fires found.")
            st.session_state['generate_clicked'] = False
    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Error: {str(e)}")
        st.session_state['generate_clicked'] = False
