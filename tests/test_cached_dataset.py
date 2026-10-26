import pandas as pd
import torch

from compass.data.datasets import CachedEmbeddingDataset


def test_cached_dataset_keys_labels_and_norm(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    torch.save({"embedding": torch.ones(6, 4, dtype=torch.float16)}, cache / "row_7.pt")
    torch.save({"embedding": torch.zeros(5, 4, dtype=torch.float16)}, cache / "row_9.pt")
    torch.save({"mean": torch.full((4,), 1.0), "std": torch.full((4,), 2.0)}, tmp_path / "norm.pt")
    pd.DataFrame({"alphafolddb-id": ["P1", "P2", "P3"], "dataset_flag": ["train", "train", "valid"],
                  "_original_row_index": [7, 9, 11], "site_labels": ["[[2], [4, 5]]", "[[1]]", "[[1]]"],
                  "label_source": ["goliath_original", "m3site", "m3site"]}).to_csv(tmp_path / "d.csv", index=False)
    ds = CachedEmbeddingDataset(str(tmp_path / "d.csv"), str(cache), "train", norm_stats=str(tmp_path / "norm.pt"))
    assert ds.protein_ids == ["P1", "P2"]
    item = ds[0]
    assert item["labels"].tolist() == [0, 1, 0, 1, 1, 0]          # range [4, 5] read inclusively
    assert torch.allclose(item["embedding"], torch.zeros(6, 4))   # (1 - 1) / 2
