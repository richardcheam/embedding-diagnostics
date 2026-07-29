import json

from jepa_lens.logging_utils import RunLogger, read_jsonl


def test_logger_roundtrips_records(tmp_path):
    logger = RunLogger(tmp_path)
    logger.log({"step": 0, "loss": 1.5})
    logger.log({"step": 500, "loss": 0.9})
    logger.close()

    records = read_jsonl(tmp_path / "metrics.jsonl")
    assert len(records) == 2
    assert records[0]["step"] == 0
    assert records[1]["loss"] == 0.9


def test_logger_appends_one_line_per_record(tmp_path):
    logger = RunLogger(tmp_path)
    for step in range(3):
        logger.log({"step": step})
    logger.close()

    lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[1])["step"] == 1


def test_logger_creates_missing_directory(tmp_path):
    nested = tmp_path / "a" / "b"
    logger = RunLogger(nested)
    logger.log({"step": 0})
    logger.close()
    assert (nested / "metrics.jsonl").exists()


def test_write_config_snapshot(tmp_path):
    logger = RunLogger(tmp_path)
    logger.write_config({"strategy": {"name": "ema_stopgrad"}})
    logger.close()

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["strategy"]["name"] == "ema_stopgrad"
