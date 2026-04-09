from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trading_platform.utils.dates import make_output_dir
from trading_platform.utils.serialization import dump_json, dump_yaml, model_to_dict


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass(slots=True)
class ExperimentRun:
    run_id: str
    run_dir: Path


class ExperimentLogger:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def start_run(self, config: Any, prefix: str = "run") -> ExperimentRun:
        run_id = f"{prefix}_{uuid.uuid4().hex[:10]}"
        run_dir = make_output_dir(self.base_dir, run_id)
        config_payload = model_to_dict(config)
        dump_yaml(run_dir / "config_snapshot.yaml", config_payload)
        dump_json(run_dir / "metadata.json", {"run_id": run_id})
        self.logger.info("Started experiment run %s in %s", run_id, run_dir)
        return ExperimentRun(run_id=run_id, run_dir=run_dir)

    def log_artifact(self, run: ExperimentRun, name: str, payload: dict[str, Any]) -> Path:
        path = run.run_dir / name
        dump_json(path, payload)
        return path
