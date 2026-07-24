import pandas as pd
from torch.utils.data import Dataset

def normalize_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy().reset_index(drop=True)
    if "Time" not in table:
        table["Time"] = pd.NaT
    table["Time"] = pd.to_datetime(table["Time"], errors="coerce", format="mixed")
    return table.sort_values("Time", kind="stable").reset_index(drop=True)


class SlidingWindowDataset(Dataset):
    """Expose every overlapping NTP window as one dataset example."""

    def __init__(
        self,
        records: Dataset,
        tensorize,
        max_table_len: int,
        stride: int | None = None,
        windows=None,
    ):
        if max_table_len < 2:
            raise ValueError("NTP max_table_len must be at least 2.")
        self.records = records
        self.tensorize = tensorize
        self.max_table_len = int(max_table_len)
        self.stride = int(stride or (max_table_len - 1))
        if self.stride <= 0:
            raise ValueError("NTP stride must be positive.")
        self.windows = list(windows) if windows is not None else []
        if windows is None:
            for record_index in range(len(records)):
                length = len(records[record_index])
                if length < 2:
                    continue
                if length <= self.max_table_len:
                    starts = [0]
                else:
                    last_start = length - self.max_table_len
                    starts = list(range(0, last_start + 1, self.stride))
                    if starts[-1] != last_start:
                        starts.append(last_start)
                self.windows.extend((record_index, start) for start in starts)
        if not self.windows:
            raise ValueError("NTP dataset contains no record with at least two events.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        record_index, start = self.windows[index]
        table = normalize_table(self.records[record_index])
        return self.tensorize(table.iloc[start : start + self.max_table_len])
__all__ = ["SlidingWindowDataset", "normalize_table"]
