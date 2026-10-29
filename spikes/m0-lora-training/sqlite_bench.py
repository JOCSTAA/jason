"""SQLite write speed in each folder given (report only; informs where the
brain's state should live once memory search arrives in M2).

    docker compose -f spikes/m0-lora-training/compose.yaml run --rm sqlite

Three tests per folder, in both of SQLite's journal modes: 500 transactions of
one row each (like memory writes during a conversation), one transaction of
20,000 rows (like consolidation during sleep), and 2,000 point reads.
The test database is deleted afterwards.
"""

import sqlite3
import statistics
import sys
import time
from pathlib import Path


def bench(folder: Path, journal: str) -> dict:
    db = folder / "bench.db"
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    con = sqlite3.connect(db, isolation_level=None)
    con.execute(f"PRAGMA journal_mode={journal}")
    con.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, text TEXT)")

    times = []
    for i in range(500):
        t = time.perf_counter()
        con.execute("BEGIN")
        con.execute("INSERT INTO entries (text) VALUES (?)", (f"fact number {i} " * 8,))
        con.execute("COMMIT")
        times.append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    con.execute("BEGIN")
    con.executemany("INSERT INTO entries (text) VALUES (?)", ((f"bulk row {i} " * 8,) for i in range(20_000)))
    con.execute("COMMIT")
    bulk_s = time.perf_counter() - t

    t = time.perf_counter()
    for i in range(1, 2001):
        con.execute("SELECT text FROM entries WHERE id = ?", (i * 7 % 20_000 + 1,)).fetchone()
    read_ms = (time.perf_counter() - t) * 1000 / 2000

    con.close()
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    return {
        "commit_ms_median": statistics.median(times),
        "commit_ms_p99": sorted(times)[int(len(times) * 0.99) - 1],
        "bulk_rows_per_s": 20_000 / bulk_s,
        "read_ms": read_ms,
    }


def main() -> None:
    print("=== sqlite_bench: paste this block ===")
    print(f"{'folder':<10}{'journal':<9}{'commit median':>15}{'commit p99':>12}{'bulk rows/s':>13}{'read':>10}")
    for arg in sys.argv[1:]:
        folder = Path(arg)
        folder.mkdir(parents=True, exist_ok=True)
        for journal in ("DELETE", "WAL"):
            r = bench(folder, journal)
            print(f"{arg:<10}{journal:<9}{r['commit_ms_median']:>12.2f} ms{r['commit_ms_p99']:>9.2f} ms"
                  f"{r['bulk_rows_per_s']:>13,.0f}{r['read_ms']:>7.3f} ms")


if __name__ == "__main__":
    main()
