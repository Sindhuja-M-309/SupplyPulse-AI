import numpy as np
from scipy.io import wavfile
import os

os.makedirs("./data/audio/firm", exist_ok=True)
os.makedirs("./data/audio/soft", exist_ok=True)

sr = 22050
t = np.linspace(0, 0.5, int(sr * 0.5))

# High pitch tone = Firm / High turgor
firm_sound = np.sin(2 * np.pi * 800 * t) * np.exp(-10 * t)
wavfile.write("./data/audio/firm/sample1.wav", sr, (firm_sound * 32767).astype(np.int16))

# Low dull frequency = Soft / Decay
soft_sound = np.sin(2 * np.pi * 200 * t) * np.exp(-5 * t)
wavfile.write("./data/audio/soft/sample1.wav", sr, (soft_sound * 32767).astype(np.int16))

print("✅ Dummy audio files created in 1 second!")