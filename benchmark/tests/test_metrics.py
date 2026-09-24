import unittest

import pandas as pd

from metrics import competition_metrics


class CompetitionMetricTest(unittest.TestCase):
    def test_parts_are_equal_weighted(self):
        rows = []
        for part in ("head", "left_body", "right_body"):
            rows.extend(
                [
                    {"part": part, "individual_id": "a", "predicted_id": "a"},
                    {"part": part, "individual_id": "b", "predicted_id": "b"},
                ]
            )
        metric = competition_metrics(pd.DataFrame(rows))
        self.assertEqual(metric["final_score"], 1.0)
        self.assertEqual(metric["mean_part_top1_accuracy"], 1.0)

    def test_missing_part_is_rejected(self):
        frame = pd.DataFrame(
            [{"part": "head", "individual_id": "a", "predicted_id": "a"}]
        )
        with self.assertRaises(ValueError):
            competition_metrics(frame)


if __name__ == "__main__":
    unittest.main()

