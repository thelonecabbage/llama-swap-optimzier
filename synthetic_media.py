"""Deterministic, dependency-free image/audio generation for benchmark cases.

These are synthetic stand-ins for real vision/audio content (no font
rendering or real speech is possible with the standard library alone). They
exist to exercise multimodal request plumbing and measure latency, not to
validate real-world perception or transcription quality.
"""

from __future__ import annotations

import io
import math
import struct
import wave
import zlib
from typing import List, Tuple

Color = Tuple[int, int, int]


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def encode_png(width: int, height: int, pixels: List[Color]) -> bytes:
    """Encode raw RGB pixels (row-major) as a minimal uncompressed-filter PNG."""
    if len(pixels) != width * height:
        raise ValueError("pixels length must equal width * height")
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type: none
        for x in range(width):
            r, g, b = pixels[y * width + x]
            raw.extend((r, g, b))
    compressed = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit depth, RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def grid_image_png(
    grid_size: int,
    cell_px: int,
    highlight_row: int,
    highlight_col: int,
    base_color: Color = (40, 40, 40),
    highlight_color: Color = (220, 30, 30),
) -> bytes:
    """Render a grid of uniform cells with exactly one highlighted cell."""
    width = height = grid_size * cell_px
    pixels: List[Color] = [base_color] * (width * height)
    for y in range(height):
        row = y // cell_px
        for x in range(width):
            col = x // cell_px
            if row == highlight_row and col == highlight_col:
                pixels[y * width + x] = highlight_color
    return encode_png(width, height, pixels)


def image_sequence_pngs(
    grid_size: int,
    cell_px: int,
    frame_positions: List[Tuple[int, int]],
    base_color: Color = (40, 40, 40),
    highlight_color: Color = (220, 30, 30),
) -> List[bytes]:
    """Render one grid image per (row, col) highlight position."""
    return [
        grid_image_png(grid_size, cell_px, row, col, base_color, highlight_color)
        for row, col in frame_positions
    ]


def tiny_probe_png() -> bytes:
    """Smallest possible image, used only to probe image-input support."""
    return encode_png(1, 1, [(0, 0, 0)])


def tone_sequence_wav(
    tone_count: int,
    tone_freq_hz: int = 880,
    tone_ms: int = 150,
    gap_ms: int = 150,
    sample_rate: int = 8000,
    amplitude: float = 0.5,
) -> bytes:
    """Render a mono 16-bit WAV with `tone_count` short sine-wave beeps."""
    tone_frames = int(sample_rate * tone_ms / 1000)
    gap_frames = int(sample_rate * gap_ms / 1000)
    frames = bytearray()
    frames.extend(b"\x00\x00" * gap_frames)
    for _ in range(tone_count):
        for i in range(tone_frames):
            sample = int(32767 * amplitude * math.sin(2 * math.pi * tone_freq_hz * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        frames.extend(b"\x00\x00" * gap_frames)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(bytes(frames))
    return buffer.getvalue()


def tiny_probe_wav() -> bytes:
    """Smallest practical WAV, used only to probe audio-input support."""
    return tone_sequence_wav(tone_count=1, tone_ms=20, gap_ms=20)
