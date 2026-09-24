import os
import unittest
from pathlib import Path

import pandas as pd


class FoldManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get("HYENA_FOLD_MANIFEST", "artifacts/folds.csv"))
        if not path.is_file():
            raise unittest.SkipTest(f"Fold manifest not built: {path}")
        cls.frame = pd.read_csv(path)

    def test_official_counts(self):
        self.assertEqual(len(self.frame), 4067)
        self.assertEqual(
            self.frame["part"].value_counts().to_dict(),
            {"head": 1963, "left_body": 1056, "right_body": 1048},
        )
        self.assertEqual(self.frame["individual_id"].nunique(), 255)

    def test_every_source_group_is_indivisible(self):
        self.assertEqual(
            int(self.frame.groupby("source_group")["fold"].nunique().max()), 1
        )

    def test_all_folds_and_rows_are_unique(self):
        self.assertEqual(sorted(self.frame["fold"].unique().tolist()), list(range(5)))
        self.assertEqual(self.frame["sample_index"].nunique(), len(self.frame))
        self.assertEqual(self.frame["image_path"].nunique(), len(self.frame))

    def test_no_avoidable_part_id_cold_start(self):
        totals = self.frame["stratum"].value_counts()
        for fold in range(5):
            train_strata = set(self.frame.loc[self.frame["fold"] != fold, "stratum"])
            valid = self.frame.loc[self.frame["fold"] == fold]
            avoidable = valid["stratum"].map(
                lambda value: value not in train_strata and totals[value] > 1
            )
            self.assertEqual(int(avoidable.sum()), 0)


if __name__ == "__main__":
    unittest.main()

