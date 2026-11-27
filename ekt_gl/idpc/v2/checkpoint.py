"""Atomic, append-only generation ledger and latest resumable state."""

from contextlib import closing
from io import BytesIO
from pathlib import Path
import pickle
import sqlite3
import zlib


def dump(value):
    return zlib.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), level=1)


def load(value):
    return pickle.loads(zlib.decompress(value))


class Checkpoint:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS trace (generation INTEGER PRIMARY KEY, row BLOB NOT NULL);
        """)
        previous = self.get("identity")
        if previous is not None and previous != identity:
            self.close()
            raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
        with self.connection:
            self.set("identity", identity)
            if previous is None:
                self.set("status", "pending")

    def get(self, key):
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else load(row[0])

    def set(self, key, value):
        self.connection.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, dump(value)))

    def generations(self):
        return (load(row) for (row,) in self.connection.execute("SELECT row FROM trace ORDER BY generation"))

    def commit(self, state, trace, model=None, result=None):
        with self.connection:
            self.connection.execute("INSERT INTO trace VALUES (?,?)", (trace["generation"], dump(trace)))
            self.set("state", state)
            if model is not None:
                self.set("model", model)
            self.set("result", result)
            self.set("status", "complete" if result is not None else "running")

    def status(self, status, error=None):
        with self.connection:
            self.set("status", status)
            self.set("error", error)

    def close(self):
        self.connection.close()


def model_state(network, torch):
    data = BytesIO()
    torch.save((network.model.state_dict(), network.optimizer.state_dict(), network.steps), data)
    return data.getvalue()


def restore_model(network, data, torch):
    model, optimizer, steps = torch.load(BytesIO(data), map_location=network.config.device,
                                          weights_only=True)
    network.model.load_state_dict(model)
    network.optimizer.load_state_dict(optimizer)
    network.steps = steps


def read_checkpoint(path, fingerprint, method, seed):
    """Read only; a reporting pass never mutates a suite checkpoint."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        metadata = {key: load(value) for key, value in db.execute(
            "SELECT key, value FROM meta WHERE key IN ('identity','status','result','error')")}
        if metadata.get("identity") != (fingerprint, method, seed):
            raise ValueError(f"Unexpected checkpoint identity {path}")
        generations = [load(row) for (row,) in db.execute("SELECT row FROM trace ORDER BY generation")]
    return metadata, generations
