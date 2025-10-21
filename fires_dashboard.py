
def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))


NOTE_SET = []
base_notes = [50, 53, 57]  # D3,F3,A3,C4
for octave in range(-1, 3):    # four octaves
    for n in base_notes:
        NOTE_SET.append(n + 12 * octave)
NOTE_SET = sorted([m for m in NOTE_SET if 36 <= m <= 84])

def map_val_to_note(v, vmin, vmax):
    """Map fire intensity to one of the D–F–A–C notes across 4 octaves."""
    if np.isnan(v):
        return NOTE_SET[len(NOTE_SET)//2]
    t = (v - vmin) / (max(1e-9, (vmax - vmin)))
    idx = int(t * (len(NOTE_SET) - 1))
    return NOTE_SET[min(max(idx, 0), len(NOTE_SET)-1)]


def one_pole_lowpass(x, sr, cutoff_hz=2500.0):
    # y[n] = y[n-1] + a * (x[n] - y[n-1]),  a = 1 - exp(-2*pi*fc/sr)
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sr)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y

def soft_env(w, sr, a_sec=0.25, r_sec=0.6):
    n = len(w)
    a = max(1, int(a_sec * sr))
    r = max(1, int(r_sec * sr))
    a = min(a, n//2); r = min(r, n - a)
    env = np.ones(n, dtype=np.float32)
    if a > 0:
        env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r, dtype=np.float32)
    return w * env


def ensemble_strings(base_hz, n, sr):
    t = np.arange(n, dtype=np.float32) / sr


    vib_rate = 5.0       # Hz
    vib_depth = 0.006


    cents = np.array([-7.0, 0.0, 7.0], dtype=np.float32)
    detune = 2.0 ** (cents / 1200.0)
    voices = np.zeros(n, dtype=np.float32)

    for d in detune:
        f_inst = base_hz * d * (1.0 + vib_depth * np.sin(2*np.pi*vib_rate*t))
        phase = 2*np.pi * np.cumsum(f_inst) / sr

        tri = (2.0/np.pi) * np.arcsin(np.sin(phase))
        voices += tri

    voices /= len(detune)


    noise = (np.random.uniform(-1.0, 1.0, n).astype(np.float32)) * 0.02
    y = voices + noise


    y = one_pole_lowpass(y, sr, cutoff_hz=2300.0)


    y = soft_env(y, sr, a_sec=0.18, r_sec=0.5)


    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak)


vmin, vmax = np.nanpercentile(fires["_val"], 5), np.nanpercentile(fires["_val"], 95)
if vmax <= vmin:
    vmax = vmin + 1.0


TARGET_DURATION_SEC = 180
global_start, global_end = fires["_start"].min(), fires["_end"].max()
total_days = (global_end - global_start).days + 1
seconds_per_day = TARGET_DURATION_SEC / total_days
samples_per_day = int(SAMPLE_RATE * seconds_per_day)
total_samples = int(SAMPLE_RATE * TARGET_DURATION_SEC)
buf = np.zeros(total_samples, dtype=np.float32)
def day_index(d): return (d - global_start).days


for _, r in fires.iterrows():
    s_idx = day_index(r["_start"]) * samples_per_day
    e_idx = (day_index(r["_end"]) + 1) * samples_per_day
    if s_idx >= len(buf):
        continue
    e_idx = min(e_idx, len(buf))
    n = e_idx - s_idx
    if n <= 0:
        continue

    midi_note = map_val_to_note(float(r["_val"]), vmin, vmax)
    freq = midi_to_hz(midi_note)
    buf[s_idx:e_idx] += 0.38 * ensemble_strings(freq, n, SAMPLE_RATE)


buf = np.tanh(buf / (np.max(np.abs(buf)) + 1e-9))


t = np.arange(len(buf), dtype=np.float32) / SAMPLE_RATE

d2 = 73.42    # D2
d3 = 146.83   # D3
d4 = 293.66# D4
a3 = 220

drone = (
    0.15 * np.sin(2 * np.pi * d2 * t) +
    0.35 * np.sin(2 * np.pi * d3 * t) +
    0.35 * np.sin(2 * np.pi * a3 * t) +
    0.15 * np.sin(2 * np.pi * d4 * t)
)


fade_len = int(2.5 * SAMPLE_RATE)  # 2.5s
fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
drone[:fade_len] *= fade_in
drone[-fade_len:] *= fade_out

# gentle breathing (amplitude) rather than vibrato (keeps pitch steady)
mod = 0.75 + 0.25 * np.sin(2 * np.pi * 0.06 * t)  # ~16.7s cycle
drone *= mod

# --- Mix drone now (after fire normalization) ---
drone *= 0.02  # make drone about 40% quieter (i.e., 60% of original loudness)
buf += drone.astype(np.float32)

# --- Softly re-normalize to avoid clipping and export ---
buf /= (np.max(np.abs(buf)) + 1e-9)
buf = np.tanh(buf)

pcm16 = (buf * 32767).astype(np.int16).tobytes()
mix = AudioSegment(
    data=pcm16, sample_width=2, frame_rate=SAMPLE_RATE, channels=1
).apply_gain(MASTER_GAIN_DB)
mix.export(OUTPUT_MP3, format="mp3")

print(f"✅ Done! Exported {OUTPUT_MP3}")
print(f"Fires: {len(fires)} | Duration: {len(mix)/1000:.1f}s (~{len(mix)/60000:.1f} min)")

