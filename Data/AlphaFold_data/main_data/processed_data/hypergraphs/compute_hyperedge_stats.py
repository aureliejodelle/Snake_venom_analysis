import json
import statistics
import sys
from pathlib import Path

base = Path(__file__).parent
output = {}
skipped = []  # (filepath, reason) for everything we couldn't read


def long_path(p: Path) -> Path:
    """On Windows, prefix with \\\\?\\ to bypass the 260-char MAX_PATH limit.
    No-op on other platforms."""
    if sys.platform == "win32":
        s = str(p.resolve())
        if not s.startswith("\\\\?\\"):
            return Path("\\\\?\\" + s)
    return p


for class_dir in sorted(base.iterdir()):
    if not class_dir.is_dir():
        continue
    summary_dir = class_dir / "summary"
    if not summary_dir.exists():
        continue

    n_hyperedges = []
    for f in sorted(summary_dir.glob("*.json")):
        try:
            with open(long_path(f), encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, OSError) as e:
            # Most common cause on Windows: path > 260 chars, or a OneDrive
            # placeholder file that never actually synced to disk.
            skipped.append((str(f), f"unreadable ({e.__class__.__name__}): {e}"))
            continue
        except json.JSONDecodeError as e:
            skipped.append((str(f), f"invalid JSON: {e}"))
            continue

        if "n_hyperedges" not in data:
            skipped.append((str(f), "missing 'n_hyperedges' key"))
            continue

        n_hyperedges.append(data["n_hyperedges"])

    if not n_hyperedges:
        continue

    output[class_dir.name] = {
        "n_proteins": len(n_hyperedges),
        "mean_hyperedges": statistics.mean(n_hyperedges),
        "median_hyperedges": statistics.median(n_hyperedges),
        "stdev_hyperedges": statistics.stdev(n_hyperedges) if len(n_hyperedges) > 1 else 0.0,
        "min_hyperedges": min(n_hyperedges),
        "max_hyperedges": max(n_hyperedges),
    }

out_path = base / "hyperedge_stats.json"
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(output, fh, indent=2)

print(f"Saved stats for {len(output)} classes to {out_path}")
for cls, stats in output.items():
    print(
        f"  {cls}: n={stats['n_proteins']}, mean={stats['mean_hyperedges']:.2f}, "
        f"median={stats['median_hyperedges']:.1f}, std={stats['stdev_hyperedges']:.2f}"
    )

if skipped:
    print(f"\nSkipped {len(skipped)} file(s) — these proteins are MISSING from the stats above:")
    for path, reason in skipped:
        print(f"  - {path}\n      {reason}")
    skipped_log = base / "hyperedge_stats_skipped.json"
    with open(skipped_log, "w", encoding="utf-8") as fh:
        json.dump([{"file": p, "reason": r} for p, r in skipped], fh, indent=2)
    print(f"\nFull skip log written to {skipped_log}")