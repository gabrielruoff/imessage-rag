"""Hermetic tests for embeddings.py's apple backend path. applefm.embed_texts_apple
is patched (Track 0 leaves it a raising stub) and the Ollama HTTPConnection is
stubbed, so no shim, no server, and no network are touched.

The two bindings under test are both module-level from-imports on embeddings, so
they are patched on the embeddings module namespace: EMBED_MODEL selects the
backend, EMBED_DIM drives _fit_dim."""

import json
import math
import unittest
from unittest import mock

import applefm
import embeddings


class TestApplePathSkipsPrefixes(unittest.TestCase):
    """With EMBED_MODEL=apple, embed_texts hands RAW text to embed_texts_apple:
    nomic's 'search_document: ' / 'search_query: ' prefixes are model-specific
    and must not leak into NLContextualEmbedding's input for either kind."""

    def test_raw_texts_passed_for_both_kinds(self):
        for kind in ("document", "query"):
            with self.subTest(kind=kind):
                fake = mock.Mock(return_value=[[0.1, 0.2, 0.3]])
                with mock.patch.object(embeddings, "EMBED_MODEL", "apple"), \
                        mock.patch.object(embeddings, "EMBED_DIM", 0), \
                        mock.patch.object(applefm, "embed_texts_apple", fake):
                    out = embeddings.embed_texts(["hello world"], kind=kind)
                # Called with the untouched text and the forwarded kind -- no prefix.
                fake.assert_called_once_with(["hello world"], kind)
                self.assertEqual(out, [[0.1, 0.2, 0.3]])


class TestApplePathAppliesFitDim(unittest.TestCase):
    """_fit_dim / the EMBED_DIM override wrap the apple vectors identically to
    the Ollama path: a smaller EMBED_DIM truncates then L2-renormalizes."""

    def test_embed_dim_truncates_and_renormalizes(self):
        # [3,4,5,6] -> first 2 = [3,4], norm 5 -> [0.6, 0.8] (unit length).
        fake = mock.Mock(return_value=[[3.0, 4.0, 5.0, 6.0]])
        with mock.patch.object(embeddings, "EMBED_MODEL", "apple"), \
                mock.patch.object(embeddings, "EMBED_DIM", 2), \
                mock.patch.object(applefm, "embed_texts_apple", fake):
            out = embeddings.embed_texts(["x"])
        (vec,) = out
        self.assertEqual(len(vec), 2)
        self.assertAlmostEqual(vec[0], 0.6)
        self.assertAlmostEqual(vec[1], 0.8)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in vec)), 1.0)


class TestNonApplePathUntouched(unittest.TestCase):
    """A non-apple EMBED_MODEL never takes the apple branch: embed_texts_apple
    stays untouched and the request goes out over the Ollama /api/embed path."""

    def setUp(self):
        # _connection caches one HTTPConnection per thread; drop any leftover so
        # the stubbed FakeConn is what this test's request is issued through.
        embeddings._thread_local.conn = None
        self.addCleanup(setattr, embeddings._thread_local, "conn", None)

    def test_ollama_path_intact_and_apple_stub_not_called(self):
        sent = {}

        class FakeResponse:
            status = 200

            def read(self):
                return b'{"embeddings": [[1.0, 2.0]]}'

        class FakeConn:
            def __init__(self, *a, **k):
                pass

            def request(self, _method, path, body=None, **_kwargs):
                sent.update(json.loads(body))

            def getresponse(self):
                return FakeResponse()

        fake_apple = mock.Mock(side_effect=AssertionError("apple branch taken"))
        with mock.patch.object(embeddings, "EMBED_MODEL", "nomic-embed-text"), \
                mock.patch.object(embeddings, "EMBED_DIM", 0), \
                mock.patch.object(applefm, "embed_texts_apple", fake_apple), \
                mock.patch.object(embeddings.http.client, "HTTPConnection", FakeConn):
            out = embeddings.embed_texts(["hi"], kind="query")
        fake_apple.assert_not_called()
        self.assertEqual(out, [[1.0, 2.0]])
        # Prefix logic still ran for the Ollama model.
        self.assertEqual(sent["input"], ["search_query: hi"])
        self.assertEqual(sent["model"], "nomic-embed-text")


class TestDetectEmbedDimRoutesThroughApple(unittest.TestCase):
    """detect_embed_dim probes via embed_texts, so EMBED_MODEL=apple routes the
    probe through embed_texts_apple; it measures the true length (here a
    non-512 mock proves it isn't hardcoded to the expected 512)."""

    def test_returns_mocked_vector_length(self):
        fake = mock.Mock(return_value=[[0.0] * 300])
        with mock.patch.object(embeddings, "EMBED_MODEL", "apple"), \
                mock.patch.object(embeddings, "EMBED_DIM", 0), \
                mock.patch.object(applefm, "embed_texts_apple", fake):
            dim = embeddings.detect_embed_dim()
        self.assertEqual(dim, 300)
        fake.assert_called_once()


if __name__ == "__main__":
    unittest.main()
