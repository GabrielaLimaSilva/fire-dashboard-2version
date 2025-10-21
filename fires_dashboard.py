def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Define note set: D-F-A (minor triad) across 4 octaves
NOTE_SET = []
base_notes = [50, 53, 57]  # D3, F3, A3
for octave in range(-1, 3):    # four octaves
    for n in base_notes:
        NOTE_SET.append(n + 12 * octave)
NOTE_SET = sorted([m for m in NOTE_SET if 36 <= m <= 84])

def map_val_to_note(v, vmin, vmax):
    """Map fire intensity to one of the D-F-A notes across 4 octaves."""
    if np.isnan(v):
        return NOTE_SET[len(NOTE_SET)//2]
    t = (v - vmin) / (max(1e-9, (vmax - vmin)))
    idx = int(t * (len(NOTE_SET) - 1))
    return NOTE_SET[min(max(idx, 0), len(NOTE_SET)-1)]

def one_pole_lowpass(x, sr, cutoff_hz=2500.0):
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
    vib_rate = 5.0
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
