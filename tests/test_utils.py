import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sonic_insight.constants import FEATURE_COLUMNS
from sonic_insight.utils import (
    generate_playlist_queries,
    nearest_neighbors,
    pseudo_feature_vector,
    score_sentiment,
)


class TestUtilsCoreLogic(unittest.TestCase):
    def test_pseudo_feature_vector_is_deterministic_and_bounded(self):
        a = pseudo_feature_vector("artist-track-seed")
        b = pseudo_feature_vector("artist-track-seed")

        self.assertEqual(a, b)

        for feature in [
            "danceability",
            "energy",
            "valence",
            "acousticness",
            "instrumentalness",
            "speechiness",
        ]:
            self.assertGreaterEqual(a[feature], 0.0)
            self.assertLessEqual(a[feature], 1.0)

        self.assertGreaterEqual(a["tempo"], 60.0)
        self.assertLessEqual(a["tempo"], 190.0)

    def test_generate_playlist_queries_fallback_shape(self):
        queries = generate_playlist_queries(
            mood_prompt="nostalgic rainy indie night",
            target_energy=0.62,
            hf_client=None,
        )

        self.assertEqual(len(queries), 12)
        self.assertIsInstance(queries[0], str)
        self.assertTrue(any("nostalgic" in q.lower() or "night" in q.lower() for q in queries))

    @patch("sonic_insight.utils.get_sentiment_pipeline", return_value=None)
    def test_score_sentiment_fallback_positive_and_negative(self, _mock_pipeline):
        pos_label, pos_score = score_sentiment("love light happy dance dream")
        neg_label, neg_score = score_sentiment("dark pain cry broken fear lost")

        self.assertEqual(pos_label, "POSITIVE")
        self.assertGreaterEqual(pos_score, 0.55)
        self.assertEqual(neg_label, "NEGATIVE")
        self.assertGreaterEqual(neg_score, 0.55)

    def test_nearest_neighbors_uses_normalized_feature_space(self):
        rows = [
            {
                "id": "a",
                "name": "A",
                "artist": "Artist A",
                "album": "Album A",
                "external_url": "",
                "preview_url": "",
                "danceability": 0.90,
                "energy": 0.90,
                "valence": 0.90,
                "tempo": 60.0,
                "acousticness": 0.10,
                "instrumentalness": 0.10,
                "speechiness": 0.10,
            },
            {
                "id": "b",
                "name": "B",
                "artist": "Artist B",
                "album": "Album B",
                "external_url": "",
                "preview_url": "",
                "danceability": 0.10,
                "energy": 0.10,
                "valence": 0.10,
                "tempo": 200.0,
                "acousticness": 0.90,
                "instrumentalness": 0.90,
                "speechiness": 0.90,
            },
        ]
        df = pd.DataFrame(rows)

        query_vector = np.array([0.90, 0.90, 0.90, 200.0, 0.10, 0.10, 0.10], dtype=np.float32)

        recs = nearest_neighbors(df, query_vector, top_k=1)

        self.assertEqual(recs.iloc[0]["id"], "a")
        self.assertTrue(set(FEATURE_COLUMNS).issubset(df.columns))


if __name__ == "__main__":
    unittest.main()
