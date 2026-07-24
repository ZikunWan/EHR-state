import os
import numpy as np
import pandas as pd
import torch.distributed as dist

from pretraining.pml import STATISTICS
from pretraining.runtime_index import ensure_runtime_index, load_ntp_windows


class Records:
    def __init__(self):
        self.read_count = 0

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.read_count += 1
        return pd.DataFrame(
            {
                "Time": pd.date_range("2020-01-01", periods=4, freq="h"),
                "Item": ["Heart rate"] * 4,
                "Value": np.arange(4) + index,
                "Unit": ["bpm"] * 4,
                "Category": ["measurement"] * 4,
            }
        )

    def metadata(self, index):
        return {"scope": "icu", "patient": f"p{index}", "diagnosis": "dx"}


def main():
    dist.init_process_group("gloo")
    shared_dir = os.environ["RUNTIME_SMOKE_DIR"]
    source = os.path.join(shared_dir, "source.csv")
    index = os.path.join(shared_dir, "runtime.sqlite")
    if dist.get_rank() == 0:
        with open(source, "w", encoding="utf-8") as file:
            file.write("source")
    dist.barrier()
    records = Records()
    specs = [
        {"item": "Heart rate", "unit": "bpm", "statistic": statistic}
        for statistic in STATISTICS
    ]
    ensure_runtime_index(
        index, {"train": records}, specs, [source], 3, 2, num_workers=2
    )
    assert len(load_ntp_windows(index, "train")) == 6
    assert records.read_count == 0, (dist.get_rank(), records.read_count)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
