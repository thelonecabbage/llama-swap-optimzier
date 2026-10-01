import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fetch_assets


class FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self._data


class FetchOneTests(unittest.TestCase):
    def _asset(self, data: bytes, filename: str = "sample.bin") -> fetch_assets.Asset:
        return fetch_assets.Asset(
            name="sample",
            repo="org/repo",
            revision="deadbeef",
            repo_path="dir/sample.bin",
            local_filename=filename,
            sha256=hashlib.sha256(data).hexdigest(),
            license="CC-BY-NC-SA-4.0",
            source_url="https://huggingface.co/datasets/org/repo",
        )

    def test_fetch_one_saves_file_when_checksum_matches(self):
        data = b"real file contents"
        asset = self._asset(data)
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            with mock.patch.object(fetch_assets.urllib.request, "urlopen", return_value=FakeResponse(data)):
                ok = fetch_assets.fetch_one(asset, force=False, out_dir=out_dir)
            self.assertTrue(ok)
            self.assertEqual((out_dir / asset.local_filename).read_bytes(), data)

    def test_fetch_one_refuses_to_save_on_checksum_mismatch(self):
        asset = self._asset(b"expected contents")
        wrong_data = b"wrong contents"
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            with mock.patch.object(fetch_assets.urllib.request, "urlopen", return_value=FakeResponse(wrong_data)):
                ok = fetch_assets.fetch_one(asset, force=False, out_dir=out_dir)
            self.assertFalse(ok)
            self.assertFalse((out_dir / asset.local_filename).exists())

    def test_fetch_one_skips_download_when_already_cached_and_verified(self):
        data = b"already here"
        asset = self._asset(data)
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / asset.local_filename).write_bytes(data)

            def fail_if_called(*args, **kwargs):
                raise AssertionError("should not re-download a verified cached file")

            with mock.patch.object(fetch_assets.urllib.request, "urlopen", side_effect=fail_if_called):
                ok = fetch_assets.fetch_one(asset, force=False, out_dir=out_dir)
            self.assertTrue(ok)

    def test_fetch_one_redownloads_when_force_is_set(self):
        data = b"fresh contents"
        asset = self._asset(data)
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / asset.local_filename).write_bytes(data)
            with mock.patch.object(fetch_assets.urllib.request, "urlopen", return_value=FakeResponse(data)) as urlopen:
                ok = fetch_assets.fetch_one(asset, force=True, out_dir=out_dir)
            self.assertTrue(ok)
            urlopen.assert_called_once()


class AssetUrlTests(unittest.TestCase):
    def test_asset_url_uses_pinned_revision(self):
        asset = fetch_assets.MANIFEST[0]
        url = fetch_assets.asset_url(asset)
        self.assertIn(asset.revision, url)
        self.assertIn(asset.repo_path, url)
        self.assertTrue(url.startswith(f"https://huggingface.co/datasets/{asset.repo}/resolve/"))


if __name__ == "__main__":
    unittest.main()
