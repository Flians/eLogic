# Experiment data publication

This directory is the compact GitHub export. Local `results_v2/` alone occupies
approximately 151 GB; the raw directories are preserved, not deleted or silently
replaced by this export. See [CURRENT.md](CURRENT.md) for the current evidence.

- `catalog.json`: studies, archive paths, SHA-256 hashes, published/local counts,
  and selection scope.
- `archives/`: gzip tar parts containing reports, commands, verification records,
  small logs, source/configuration snapshots, and the selected V9 final netlists
  and independent CEC artifacts. Paths inside archives are repository-relative.
- `contents.jsonl.gz`: SHA-256 and archive location of every exported file.
- `local_inventory.jsonl.gz`: local file sizes and publication decisions, plus
  explicitly excluded subtrees. This is a size inventory, not a hash proof of
  unexported raw data.
- `assets/`: lossless gzip stream parts of the current local legacy
  `lib_expr2cost.json`; concatenate parts in catalog order before decompressing.

Selection is independent of success/failure and quality. Build caches, copied
third-party source trees, large intermediate networks and serialized graphs are
not all uploaded. Size limits also apply uniformly; the catalog and inventory
make omissions explicit. Do not describe this as a complete raw-data mirror or
as complete EPFL evaluation. Existing Git-tracked ePMapping evidence remains in
its recorded locations.

Validate the published containers and every archived member:

```sh
python3 scripts/maintenance/verify_published_data.py --deep
```

To inspect a study, find its parts in `catalog.json`, then extract those parts
into a separate empty directory, preserving their repository-relative paths:

```sh
mkdir -p /tmp/migbalance-evidence
tar -xzf data/archives/STUDY.part001.tar.gz -C /tmp/migbalance-evidence
```

Records retain original command arguments, paths, hashes, timeouts and failures.
Some old records name absolute paths or binaries not shipped here. Do not edit
their frozen provenance to claim portability: build/replay in a new explicit
checkpoint and record any dependency changes.

The legacy database's compressed copy is separate from the older Git LFS object
at the repository root. This export does not rewrite Git history or pretend the
older object has changed. To reconstruct the new database, concatenate the
catalog's `database.parts` in order, run `gzip -dc`, and verify the resulting
`database.sha256` before use. The ordinary eGraphMap/ePMapping workflows use
their recorded libraries and inputs rather than this legacy database.
