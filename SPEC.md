# Speaker Measure — Project Specification

## 1. Concept & Vision

A professional-grade home cinema room measurement tool for Windows. Plays lossless test signals (sweeps) through the AV receiver via ASIO, captures the response via measurement microphone, processes the impulse response, and exports 48 kHz WAV files compatible with REW and other calibration tools.

Feel: **clinical precision meets dark-mode studio aesthetic.** Not a toy — a serious measurement instrument that happens to have a UI.

---

## 2. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.10+ | Ecosystem, audio libs |
| **Audio I/O** | `sounddevice` (PortAudio + ASIO) | Low-latency ASIO on Windows, playback+capture combined |
| **GUI** | PyQt6 | Mature, professional look, good widget set |
| **Signal Processing** | numpy + scipy | FFT, deconvolution, filtering |
| **WAV I/O** | soundfile | Lossless FLAC/PCM WAV read/write |
| **Packaging** | PyInstaller | Windows .exe generation |

**Why not C++/JUCE?** Python iteration speed is 10× faster for a tool of this complexity. ASIO latency via PortAudio is already excellent (~1-2 ms with small buffers).

---

## 3. Core Architecture

### 3.1 Measurement Pipeline

```
1. User configures: ASIO playback device + mic input
2. App plays exponential sine sweep (20 Hz → 20 kHz, ~3 sec)
3. App captures mic response (simultaneous play+rec via sounddevice.playrec)
4. Deconvolve impulse response: IR = inverse filter × captured response
5. Window and smooth IR → frequency response
6. Export as 48 kHz WAV + generate target curve
```

### 3.2 Multi-Channel Support

- Query all ASIO-available channels from selected device
- Support layouts: stereo → 5.1 → 7.1 → 9.4.6 (all speaker pairs measured sequentially)
- Subwoofers measured individually

### 3.3 Key Modules

```
src/
  audio_engine.py      # sounddevice playrec, ASIO device handling, buffer management
  signal_processor.py  # sweep generation, deconvolution, IR windowing, FFT
  hdmi_channel_detector.py  # HDMI channel enumeration, subwoofer count, layout detection
  avr_control.py       # AVR Telnet control (bass mode, subwoofer switching)
  measurement.py       # per-channel measurement + subwoofer switching orchestration
  mic_calibration.py   # apply mic calibration curve (optional .cal file)
  exporter.py          # WAV export, delay/volume offset calculation
  ui/
    main_window.py     # PyQt6 main window
    device_panel.py    # ASIO device + mic selection
    measure_view.py    # Live measurement display
    results_view.py    # Per-channel results, averaging, export
```

---

## 4. Feature Breakdown

### Phase 1 — Core Audio Engine
- [ ] Enumerate ASIO devices (show sample rate, channel count)
- [ ] Playback stereo WAV files via selected ASIO device
- [ ] Simultaneous play+rec (`sounddevice.playrec`)
- [ ] Configurable buffer size (256 / 512 / 1024 / 2048 samples)
- [ ] Target latency: <5 ms with ASIO

### Phase 2 — Measurement Signal & Processing
- [ ] Generate exponential sine sweep (20 Hz → 20 kHz, 3 sec)
- [ ] Deconvolution (inverse filter in frequency domain)
- [ ] IR windowing (time-gating to remove reflections)
- [ ] Frequency response calculation (magnitude FFT)
- [ ] Apply mic calibration .cal file (if provided)

### Phase 3 — Multi-Channel Measurement
- [x] Sequential channel measurement (FL → C → FR → etc.)
- [x] Subwoofer detection (low-frequency dominance)
- [x] Multi-position averaging (up to 16 positions)
- [x] Per-position capture → spatial average → single IR per channel
- [x] **HDMI channel detection**: auto-detect all HDMI/ASIO output channels and map to standard labels (FL, FR, C, LFE, SW1/SW2/SW3/SW4, FDL, FDR, etc.)
- [x] **Subwoofer switching prompt**: when 2+ subwoofers are configured, pause and display a message instructing the user to switch subwoofers ON/OFF before each sub measurement
- [x] **Selected-speaker measurement**: measure a specific subset of speakers (e.g. FDL, FDR, SW1, SW2) instead of all channels
- [x] **REW API import**: play sweep WAV + per-channel .mpl files through AVR, record UMIK-1 response, import into REW for deconvolution and frequency response. AVR stays in Dolby Atmos mode throughout.

### Phase 4 — Results & Export
- [ ] Display frequency response per channel (plot)
- [ ] Export 48 kHz WAV per channel (REW-compatible)
- [ ] Calculate required delay offsets (group delay at crossover)
- [ ] Calculate volume level offsets (relative to reference)
- [ ] Summary table: channel | delay_ms | level_offset_db

### Phase 5 — Mic Centering & Utilities
- [ ] Automated mic centering tool (phase correlation sweep)
- [ ] Mic calibration file loader (UMM format or plain text)
- [ ] Export all results as ZIP (WAV files + summary CSV)

---

## 5. UI Design

### Color Palette (Dark Studio Theme)
- Background: `#0f0f0f`
- Panel: `#1a1a1a`
- Border: `#2a2a2a`
- Primary accent: `#00d4ff` (cyan — measurement trace)
- Secondary accent: `#ff6b35` (orange — target/reference)
- Text: `#e0e0e0`
- Disabled: `#666666`

### Layout
```
┌─────────────────────────────────────────────────────────┐
│  [Logo] Speaker Measure          [Device: ASIO    ▼]   │
├──────────────────┬────────────────────────────────────┤
│                  │                                     │
│  DEVICE PANEL    │         MEASUREMENT VIEW           │
│  - Playback dev  │    [Live frequency response plot]   │
│  - Mic input     │                                     │
│  - Sample rate   │                                     │
│  - Buffer size   │                                     │
│                  │                                     │
│  [Calibration]   │                                     │
│  [Configure]     │                                     │
│                  ├────────────────────────────────────┤
│  CHANNEL LIST    │         RESULTS TABLE               │
│  ☑ FL  ☑ C  ☑ FR│   Ch   Delay    Level   WAV        │
│  ☑ SW1 ☑ SW2    │   FL   12.3ms   -1.2dB  [export]   │
│  [Measure All]   │   C    0ms       0dB    [export]   │
│                  │                                     │
└──────────────────┴────────────────────────────────────┘
```

---

## 6. Audio Signal Design

### Exponential Sine Sweep
- Duration: 3 seconds
- Start frequency: 20 Hz
- End frequency: 20,000 Hz
- Amplitude: -12 dBFS (protect against clipping)
- Mathematical form: `sin(2π * f(t) * t)` where `f(t) = f_start * exp(t * ln(f_end/f_start) / T)`

### Deconvolution
```
Sweep spectrum: S(f)
Captured signal: C(f)
Inverse filter: H(f) = 1 / S(f) (with safety floor)
IR(f) = C(f) * H(f)
IR(t) = inverse_fft(IR(f))
```

### IR Windowing
- Time-gate: 0 ms to first reflection (~20 ms for typical room)
- Apply cosine window at tail to avoid truncation artifacts

---

## 7. File Formats

### Export: Per-Channel WAV
- Format: 48 kHz, 24-bit PCM, stereo
- Filename: `{channel_id}_{timestamp}.wav`
- Contains: impulse response as audio samples

### Mic Calibration File (optional)
- Format: plain text, two columns `freq_hz  spl_db_offset`
- Applied as frequency-domain correction to raw measurement

### Results CSV
```csv
channel,delay_ms,level_offset_db,wav_file
fl,12.3,-1.2,fl_20260501_133045.wav
c,0.0,0.0,c_20260501_133045.wav
...
```

---

## 8. Open Questions / Decisions Needed

1. **ASIO driver availability**: User must install ASIO driver for their audio interface. App should detect and show friendly error if no ASIO device found.
2. **Mic detection**: How does user identify which input is the measurement mic vs. monitor mix? Show channel labels.
3. **Calibration file format**: Support a standard (e.g., REW's `.cal` format or plain UMM)?
4. **AVR integration**: GSonic uses TrueHD HDMI passthrough — does Vasu need HDMI/TrueHD decoding, or is the AVR already decoding to multichannel PCM via ASIO?
5. **Measurement trigger**: Manual start or automatic when level threshold exceeded?

---

## 9. Development Phases

```
Phase 1: Audio engine (device enum, playback, capture)     ← start here
Phase 2: Sweep generation + IR deconvolution
Phase 3: Per-channel measurement + results display
Phase 4: Multi-position averaging
Phase 5: Export + offset calculation + mic centering
```