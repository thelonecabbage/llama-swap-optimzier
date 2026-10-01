import io
import struct
import unittest
import wave

import synthetic_media


class PngTests(unittest.TestCase):
    def test_encode_png_rejects_mismatched_pixel_count(self):
        with self.assertRaisesRegex(ValueError, "pixels length"):
            synthetic_media.encode_png(2, 2, [(0, 0, 0)])

    def test_encode_png_header_declares_requested_dimensions(self):
        png = synthetic_media.encode_png(3, 2, [(0, 0, 0)] * 6)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        # IHDR chunk: 4-byte length, "IHDR", width(4) height(4) ...
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (3, 2))

    def test_grid_image_png_is_deterministic(self):
        first = synthetic_media.grid_image_png(4, 4, 1, 2)
        second = synthetic_media.grid_image_png(4, 4, 1, 2)
        self.assertEqual(first, second)

    def test_grid_image_png_dimensions_match_grid_size_times_cell_px(self):
        png = synthetic_media.grid_image_png(4, 6, 0, 0)
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (24, 24))

    def test_image_sequence_pngs_returns_one_frame_per_position(self):
        frames = synthetic_media.image_sequence_pngs(4, 4, [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(frame.startswith(b"\x89PNG\r\n\x1a\n") for frame in frames))

    def test_tiny_probe_png_is_1x1(self):
        png = synthetic_media.tiny_probe_png()
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (1, 1))


class WavTests(unittest.TestCase):
    def test_tone_sequence_wav_is_valid_and_mono_16bit(self):
        wav_bytes = synthetic_media.tone_sequence_wav(tone_count=2, tone_ms=10, gap_ms=10, sample_rate=8000)
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getsampwidth(), 2)
            self.assertEqual(reader.getframerate(), 8000)
            self.assertGreater(reader.getnframes(), 0)

    def test_tone_sequence_wav_is_deterministic(self):
        first = synthetic_media.tone_sequence_wav(tone_count=3)
        second = synthetic_media.tone_sequence_wav(tone_count=3)
        self.assertEqual(first, second)

    def test_tiny_probe_wav_is_short_and_valid(self):
        wav_bytes = synthetic_media.tiny_probe_wav()
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            duration_s = reader.getnframes() / reader.getframerate()
        self.assertLess(duration_s, 0.5)


if __name__ == "__main__":
    unittest.main()
