"""Small stdlib-only I/O helpers for the two priority pilots."""
import csv
import hashlib
import json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

class CSV:
    def __init__(self, path):
        self.file = Path(path).open("x", newline="", encoding="utf-8")
        self.writer = None
    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
    def close(self):
        self.file.close()
