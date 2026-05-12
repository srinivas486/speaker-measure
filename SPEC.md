# Speaker Measure — Project Specification

## 1. Concept & Vision

A professional-grade home cinema room measurement tool for Windows and macOS. Plays lossless test signals through the AV receiver (ASIO on Windows, CoreAudio on macOS), captures the response via measurement microphone, and imports recorded sweeps into REW for deconvolution and room correction analysis.

Feel: **clinical precision meets dark-mode studio aesthetic.** Not a toy — a serious measurement instrument that happens to have a UI.

**Three measurement modes:**
1. **REW workflow**: Play pre-recorded TrueHD sweep files (.mpl) through the AVR, record UMIK-1 response, import into REW for processing. AVR stays in Dolby Atmos mode throughout.
2. **Internal sweep-file mode** *(recommended for Atmos, no REW required)*: Same as REW workflow but uses the built-in deconvolution instead of REW. Load the .mpl TrueHD file once as sweep reference; all Atmos speakers (TFL, TFR, FHL, FHR, etc.) play that sweep routed to their respective HDMI channels, then deconvolved against the same reference. Works on Windows + macOS with no REW needed.
3. **Internal sweep mode**: Generate sine sweeps internally, play through AVR, capture and deconvolve locally (Phase 2 signal processing pipeline). Best for simple non-Atmos setups.

---

## 2. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.10+ | Ecosystem, audio libs |
| **Audio I/O** | `sounddevice` (PortAudio + ASIO/CoreAudio) | Low-latency ASIO on Windows, CoreAudio on macOS, playback+capture combined |
| **GUI** | PyQt6 | Mature, professional look, good widget set |
| **Signal Processing** | numpy + scipy | FFT, deconvolution, filtering |
| **WAV I/O** | soundfile | Lossless FLAC/PCM WAV read/write |
| **External tools** | `ffmpeg` (TrueHD MLP decode), REW HTTP API | MLP file decoding + REW import/deconvolution |

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
  audio_engine.py         # sounddevice playrec, ASIO/CoreAudio device handling, buffer management
  signal_processor.py      # sweep generation, deconvolution, IR windowing, FFT
  hdmi_channel_detector.py # HDMI channel enumeration, subwoofer count, layout detection
  avr_control.py          # AVR Telnet control (bass mode, subwoofer switching)
  measurement.py          # per-channel measurement + subwoofer switching orchestration
  mic_calibration.py     # apply mic calibration curve (optional .cal file)
  exporter.py            # WAV export, delay/volume offset calculation
  rew_api.py             # HTTP API client for REW (localhost:4735)
  rew_measurement.py     # REW workflow orchestrator (TrueHD .mpl + sweep.wav)
  ui/
    main_window.py        # PyQt6 main window
    device_panel.py      # ASIO device + mic selection
    measure_view.py       # Live measurement display
    results_view.py      # Per-channel results, averaging, export
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
- [x] **Internal sweep-file Atmos mode**: Load a .mpl TrueHD or .wav sweep file as the reference sweep. All Atmos/non-Atmos speakers are measured by routing this sweep to their HDMI output and deconvolving against the same reference. No REW required. Supports .mpl (via ffmpeg decode) and .wav/.flac/.aiff (via soundfile). Configured via `config.sweep_file`.
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

## 8. REW Measurement Workflow

### File types
| Extension | Description |
|---------|-------------|
| `.mpl` | TrueHD Meridian Lossless file (8-channel, per-channel sweep — decoded via `ffmpeg`) |
| `.wav` | REW reference sweep stimulus (stereo Float32, 48kHz, 12.29s, -12dBFS) |
| `.flac` | FLAC lossless (future use) |
| `.aiff` | AIFF lossless (future use) |

### TrueHD MLP decoding
MLP files (`.mpl`) are Meridian Lossless Packing containers with 8 audio channels. Each file has **one active channel** (the signal) with all other channels silent. Decoding is done via `ffmpeg` subprocess:
```bash
ffmpeg -hide_banner -loglevel error \
  -i channel.mlp \
  -map 0:a:0 \
  -af aformat=sample_fmts=fltp:channel_layouts=stereo \
  -ar 48000 -f f32le pipe:1
```

### Subwoofer switching workflow
For multi-subwoofer systems, the app pauses before each subwoofer measurement and displays:
```
===========================
  SUBWOOFER SWITCH REQUIRED
===========================
  Measure : SW2
  Switch ON : SW2
  Switch OFF: SW1, SW3, SW4
===========================
  Press ENTER after switching subwoofers to continue...
```


## 9. Open Questions / Decisions Needed

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