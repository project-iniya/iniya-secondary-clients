"""
AudioMixin.py — STT + TTS for Iniya

Expects self.profile (IniyaProfile) to already be set.
Call self.setup_speech() once after __init__.

STT:  audio buffer (bytes, numpy array, or file path)  →  str
TTS:  str  →  bytes (raw PCM, 16-bit mono, 22050 Hz)
      or plays directly via sounddevice
"""

import io
import json
import queue
import subprocess
import tempfile
import threading
import wave
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────────────────────────────────────

AudioInput  = Union[bytes, np.ndarray, str, Path]   # buffer, array, or file path
OnPartial   = Callable[[str], None]                 # streaming STT callback

# ─────────────────────────────────────────────────────────────────────────────
#  Mixin
# ─────────────────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return audio
    ratio      = to_rate / from_rate
    new_length = int(len(audio) * ratio)
    indices    = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio)


class AudioMixin:
    """
    Add to your client class:

        class IniyaClient(AudioMixin):
            def __init__(self):
                self.profile = get_profile()
                ensure_models(self.profile)
                self.setup_speech()
    """

    # ── setup ──────────────────────────────────────────────────────────────

    def setup_speech(self) -> None:
        """
        Loads STT model into memory.
        Piper (TTS) is a subprocess — no preloading needed.
        """
        self._stt_engine = self.profile.stt["engine"]
        self._stt_live_engine = self.profile.stt_live["engine"]  # always "vosk"
        self._tts_engine = self.profile.tts["engine"]

        self._vosk_rec   = None
        self._whisper    = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop   = threading.Event()

        self._load_vosk()

        if self._stt_engine == "whisper":
            self._load_whisper()

        self.device_index = None
        self.SAMPLE_RATE = 16000   # default, may be overridden by set_input_device()

    def _load_vosk(self) -> None:
        from vosk import Model, KaldiRecognizer
        model_path = self.profile.stt_model_path()
        print(f"[STT] Loading Vosk from {model_path.name}...")
        self._vosk_model = Model(str(model_path))
        # KaldiRecognizer is created fresh per call (stateless decode)
        self._VoskRecognizer = KaldiRecognizer
        print("[STT] Vosk ready.")

    def _load_whisper(self) -> None:
        import whisper
        stt = self.profile.stt
        print(f"[STT] Loading whisper {stt['model']} on {stt['device']}...")
        self._whisper = whisper.load_model(
            stt["model"],
            device=stt["device"],
            download_root=str(self.profile.whisper_cache()),
        )
        print("[STT] Whisper ready.")


    # ─────────────────────────────────────────────────────────────────────────
    #  STT — transcribe
    # ─────────────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: AudioInput,
        sample_rate: int = 16000,
        language: str = "en",
        on_partial: Optional[OnPartial] = None,
    ) -> str:
        """
        Convert audio → text.

        Args:
            audio:       bytes (raw PCM int16) | np.ndarray (float32/int16)
                         | str/Path (wav file path)
            sample_rate: only used when audio is bytes or ndarray
            language:    language hint (whisper only)
            on_partial:  optional callback(partial_text) — Vosk only,
                         called as words come in during a long buffer

        Returns:
            Transcribed string (stripped, lowercase if Vosk).
        """
        raw = self._to_pcm_bytes(audio, sample_rate)

        if self._stt_engine == "vosk":
            return self._transcribe_vosk(raw, sample_rate, on_partial)
        elif self._stt_engine == "whisper":
            return self._transcribe_whisper(raw, sample_rate, language)
        else:
            raise RuntimeError("No STT engine loaded. Check device_profile warnings.")

    def _to_pcm_bytes(self, audio: AudioInput, sample_rate: int) -> bytes:
        """Normalise any input type → raw int16 PCM bytes."""
        if isinstance(audio, (str, Path)):
            with wave.open(str(audio), "rb") as wf:
                return wf.readframes(wf.getnframes())

        if isinstance(audio, np.ndarray):
            if audio.dtype != np.int16:
                # float32 [-1,1] → int16
                audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            return audio.tobytes()

        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)

        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _transcribe_vosk(
        self,
        raw: bytes,
        sample_rate: int,
        on_partial: Optional[OnPartial],
    ) -> str:
        rec = self._VoskRecognizer(self._vosk_model, float(sample_rate))
        rec.SetWords(True)

        chunk_size = sample_rate * 2  # 0.5 s chunks
        result_parts: list[str] = []

        for i in range(0, len(raw), chunk_size):
            chunk = raw[i : i + chunk_size]
            if rec.AcceptWaveform(chunk):
                text = json.loads(rec.Result()).get("text", "").strip()
                if text:
                    result_parts.append(text)
            else:
                if on_partial:
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    if partial:
                        on_partial(partial)

        # Flush final
        final = json.loads(rec.FinalResult()).get("text", "").strip()
        if final:
            result_parts.append(final)

        return " ".join(result_parts)

    def _transcribe_whisper(
        self,
        raw: bytes,
        sample_rate: int,
        language: str,
    ) -> str:
        # openai-whisper expects float32 [-1, 1] at 16kHz
        audio_f32 = (
            np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # boost gain before resampling
        audio_f32 = np.clip(audio_f32 * 5.0, -1.0, 1.0)   # 5x gain, tweak as needed

        # whisper always needs 16kHz — resample if device runs at different rate
        if sample_rate != 16000:
            audio_f32 = _resample(audio_f32, sample_rate, 16000)

        result = self._whisper.transcribe(
            audio_f32,
            language=language,
            fp16=self.profile.stt["device"] == "cuda",
        )

        return result["text"].strip()


    # ─────────────────────────────────────────────────────────────────────────
    #  TTS — synthesise
    # ─────────────────────────────────────────────────────────────────────────

    def synthesise(
        self,
        text: str,
        play: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """
        Convert text → audio.

        Args:
            text:        the string to speak
            play:        if True, plays audio via sounddevice immediately
            output_path: if set, also saves a .wav file to this path

        Returns:
            Raw int16 PCM bytes (22050 Hz, mono).
        """
        if self._tts_engine != "piper":
            raise RuntimeError(f"Unsupported TTS engine: {self._tts_engine}")

        return self._synthesise_piper(text, play=play, output_path=output_path)

    # alias — some people say speak(), some say synthesise()
    def speak(self, text: str, play: bool = True) -> bytes:
        return self.synthesise(text, play=play)

    def _synthesise_piper(
        self,
        text: str,
        play: bool,
        output_path: Optional[Union[str, Path]],
    ) -> bytes:
        piper_bin   = str(self.profile.tts_bin_path())
        voice_model = str(self.profile.tts_voice_path())
        voice_config = voice_model + ".json"

        cmd = [
            piper_bin,
            "--model",        voice_model,
            "--config",       voice_config,
            "--output-raw",                   # raw PCM to stdout
            "--sentence-silence", "0.3",      # small pause between sentences
        ]

        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Piper failed:\n{proc.stderr.decode(errors='replace')}"
            )

        raw_pcm = proc.stdout   # int16, mono, 22050 Hz

        if output_path is not None:
            self._save_wav(raw_pcm, Path(output_path), sample_rate=22050)

        if play:
            self._play_pcm(raw_pcm, sample_rate=22050)

        return raw_pcm

    # ── audio helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _save_wav(raw_pcm: bytes, path: Path, sample_rate: int = 22050) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)       # int16 = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(raw_pcm)

    @staticmethod
    def _play_pcm(raw_pcm: bytes, sample_rate: int = 22050) -> None:
        try:
            import sounddevice as sd
            audio = np.frombuffer(raw_pcm, dtype=np.int16)
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except ImportError:
            # Fall back to writing a temp wav and opening it
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                AudioMixin._save_wav(raw_pcm, Path(f.name), sample_rate)
                tmp = f.name
            os.startfile(tmp)   # Windows: opens default audio player


    # ─────────────────────────────────────────────────────────────────────────
    #  Live mic streaming  (optional — call start/stop)
    # ─────────────────────────────────────────────────────────────────────────

    def get_device_by_index(self, idx: int) -> Optional[dict]:
        """Helper to get device info dict by index, or None if invalid."""
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("pip install sounddevice")
        try:
            dev = sd.query_devices(idx)
            if dev["max_input_channels"] <= 0:
                raise ValueError("Device is not an input device")

            return {
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "samplerate": dev["default_samplerate"],
                "hostapi": sd.query_hostapis(dev["hostapi"])["name"]
            }
        except Exception as e:
            return None


    def set_input_device(self, index: int):
        self.device_index = index
        dev = self.get_device_by_index(index)
        if dev:
            print(f"Input device set to: {dev['name']}")
            self.SAMPLE_RATE = int(dev["samplerate"])
        else:
            print(f"Invalid input device index: {index}")


    def list_usable_input_devices(self):
            
        try:
              import sounddevice as sd
        except ImportError:
              raise RuntimeError("pip install sounddevice")
        usable = []

        for idx, dev in enumerate(sd.query_devices()):
            name = dev["name"].lower()

            # must be input
            if dev["max_input_channels"] <= 0:
                continue

            # must support Whisper settings
            try:
                sd.check_input_settings(
                    device=idx,
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype="float32"
                )
            except Exception:
                continue

            usable.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "samplerate": dev["default_samplerate"]
            })

        return usable

    def start_listening(
        self,
        on_result: Callable[[str], None],
        on_partial: Optional[OnPartial] = None,
    ) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("pip install sounddevice")

        sample_rate = self.SAMPLE_RATE
        self._stream_stop.clear()
        audio_q: queue.Queue[bytes] = queue.Queue()

        def _mic_callback(indata, frames, time, status):
            audio_q.put(bytes(indata))

        # ── Vosk path — word-by-word streaming ──────────────────────────────
        def _worker_vosk():
            rec = self._VoskRecognizer(self._vosk_model, 16000.0)
            chunk_count = 0

            with sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=_mic_callback,
                device=self.device_index,
            ):
                while not self._stream_stop.is_set():
                    try:
                        data = audio_q.get(timeout=0.5)
                    except queue.Empty:
                        print("[STT] queue empty...")   # ← are we even getting audio?
                        continue

                    chunk_count += 1

                    if sample_rate != 16000:
                        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        data = _resample(audio_np, sample_rate, 16000).astype(np.int16).tobytes()

                    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    audio_np = np.clip(audio_np * 5.0, -32768, 32767)   # 5x gain, tweak as needed
                    data = audio_np.astype(np.int16).tobytes()

                    accepted = rec.AcceptWaveform(data)

                    if accepted:
                        text = json.loads(rec.Result()).get("text", "").strip()
                        print(f"[STT] result: '{text}'")
                        if text:
                            on_result(text)
                    else:
                        partial = json.loads(rec.PartialResult()).get("partial", "")
                        if on_partial and partial:
                            on_partial(partial)

            print("[STT] loop exited, flushing final...")
            final = json.loads(rec.FinalResult()).get("text", "").strip()
            print(f"[STT] final: '{final}'")
            if final:
                on_result(final)

        worker = _worker_vosk

        self._stream_thread = threading.Thread(target=worker, daemon=True)
        self._stream_thread.start()
        print(f"[STT] Mic stream started ({self._stt_engine}).")

    def stop_listening(self) -> None:
        self._stream_stop.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=10)  # was 2 — too short
        print("[STT] Mic stream stopped.")