"""Setup A: macOS voice processing (Apple's echo cancellation) through PyObjC.

Apple cancels echo only when capture and playback run in the same
AVAudioEngine, because the engine's own output is the echo reference. So one
engine plays Jason's audio through a player node and records the mic through a
tap on the input node, with voice processing turned on for the input node.

Known traps (forum reports): attach the playback graph before enabling voice
processing; the processed input can have several channels (only the first
carries the voice); mic and speaker must be the same device.
"""

import threading
import time

import AVFoundation as AVF
import numpy as np

from common import resample


def _fail(what, ok_err):
    ok, err = ok_err if isinstance(ok_err, tuple) else (ok_err, None)
    if not ok:
        raise RuntimeError(f"{what} failed: {err}")


class VoiceProcessingIO:
    def __init__(self, voice_processing: bool = True, play_sr: int = 48_000):
        self.engine = AVF.AVAudioEngine.alloc().init()
        self.player = AVF.AVAudioPlayerNode.alloc().init()
        self.play_sr = play_sr
        self.play_fmt = AVF.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(play_sr, 1)
        self.engine.attachNode_(self.player)
        self.engine.connect_to_format_(self.player, self.engine.mainMixerNode(), self.play_fmt)

        inp = self.engine.inputNode()
        if voice_processing:
            _fail("setVoiceProcessingEnabled", inp.setVoiceProcessingEnabled_error_(True, None))
        self.in_fmt = inp.outputFormatForBus_(0)
        self.in_sr = int(self.in_fmt.sampleRate())
        self.in_channels = int(self.in_fmt.channelCount())

        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self.callback_frames: list[int] = []  # buffer sizes the tap delivers (adds latency)

        def tap(buf, when):
            n = int(buf.frameLength())
            ch0 = buf.floatChannelData()[0]
            x = np.frombuffer(ch0.as_buffer(n), dtype=np.float32).copy()
            with self._lock:
                self._chunks.append(x)
                self.callback_frames.append(n)

        self._tap = tap
        # bufferSize is a request; macOS often delivers bigger buffers.
        inp.installTapOnBus_bufferSize_format_block_(0, 480, self.in_fmt, tap)

    def start(self):
        _fail("engine start", self.engine.startAndReturnError_(None))

    def stop(self):
        self.player.stop()
        self.engine.inputNode().removeTapOnBus_(0)
        self.engine.stop()
        try:  # Disable before tearing down; enabled-at-teardown has crashed for others.
            self.engine.inputNode().setVoiceProcessingEnabled_error_(False, None)
        except Exception:
            pass

    def play(self, audio: np.ndarray, sr: int):
        x = resample(audio, sr, self.play_sr)
        buf = AVF.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(self.play_fmt, len(x))
        buf.setFrameLength_(len(x))
        np.frombuffer(buf.floatChannelData()[0].as_buffer(len(x)), dtype=np.float32)[:] = x
        self.player.scheduleBuffer_completionHandler_(buf, None)
        self.player.play()

    def recorded(self) -> tuple[np.ndarray, int]:
        with self._lock:
            x = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        return x, self.in_sr


if __name__ == "__main__":
    # Probe: does voice processing switch on from Python, and what comes out?
    io = VoiceProcessingIO()
    io.start()
    time.sleep(1.5)
    x, sr = io.recorded()
    io.stop()
    frames = io.callback_frames
    print(f"input format: {sr} Hz, {io.in_channels} channel(s)")
    print(f"recorded {len(x) / sr:.2f} s, level {20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-10):.1f} dBFS")
    print(f"tap buffers: {len(frames)} callbacks, frames per callback {sorted(set(frames))}")
