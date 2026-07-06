from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

_OHLCV = ["open", "high", "low", "close", "volume"]


class ParquetCache:
    """Partitioned parquet store of OHLCV bars.

    Layout (Hive-partitioned by symbol then year):
        root/symbol=AAPL/year=2024/part-0.parquet

    Reads use pyarrow dataset filters, so a query for one symbol/date-range
    prunes whole partitions (symbol, year) and skips row-groups whose stats
    fall outside the range — i.e. predicate pushdown, not a full scan.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write(self, symbol: str, bars: pd.DataFrame) -> None:
        """Append `bars` (datetime-indexed OHLCV) into the partitioned tree."""
        if bars.empty:
            return
        frame = bars.rename_axis("date").reset_index()
        frame["symbol"] = symbol
        frame["year"] = frame["date"].dt.year

        table = pa.Table.from_pandas(frame, preserve_index=False)
        ds.write_dataset(
            table,
            self.root,
            format="parquet",
            partitioning=["symbol", "year"],
            partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
        )

    def read(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return cached bars for `symbol` in [start, end], or empty if none.

        The filter is pushed down into the scan (partition + row-group pruning).
        """
        if not self.root.exists():
            return pd.DataFrame(columns=_OHLCV)

        dataset = ds.dataset(self.root, format="parquet", partitioning="hive")
        if "symbol" not in dataset.schema.names:
            return pd.DataFrame(columns=_OHLCV)  # nothing written yet

        start, end = pd.Timestamp(start), pd.Timestamp(end)
        predicate = (
            (ds.field("symbol") == symbol)
            & (ds.field("date") >= pa.scalar(start))
            & (ds.field("date") <= pa.scalar(end))
        )
        table = dataset.to_table(filter=predicate)
        if table.num_rows == 0:
            return pd.DataFrame(columns=_OHLCV)

        out = (
            table.to_pandas()
            .drop(columns=["symbol", "year"], errors="ignore")
            .set_index("date")
            .sort_index()
        )
        return out[_OHLCV]
