"""Lightweight IOH logger that records (evaluations, raw_y, x0..xd) per call."""

import pandas as pd
from ioh import LogInfo, logger


class TrajectoryLogger(logger.AbstractLogger):
    """Captures every evaluation into a list of records for behaviour analysis."""

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.records = []

    def __call__(self, log_info: LogInfo):
        row = {
            "evaluations": log_info.evaluations,
            "raw_y": log_info.raw_y,
        }
        coords = log_info.x[: self.dim]
        for i, v in enumerate(coords):
            row[f"x{i}"] = v
        self.records.append(row)

    def reset(self, func):
        super().reset()
        self.records = []

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
