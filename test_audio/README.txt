Place your test audio files here.

Expected files:
  noisy.wav    ← your noisy input recording (16 kHz, mono, any length)
  clean.wav    ← matching clean reference (optional, needed only for metrics)

Run:
  python enhance.py --input test_audio/noisy.wav --output test_audio/enhanced.wav
  python enhance.py --input test_audio/noisy.wav --output test_audio/enhanced.wav --clean test_audio/clean.wav
