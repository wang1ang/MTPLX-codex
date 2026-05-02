# Changelog

All notable user-facing changes are recorded here.

## v0.1.0-preview

### Added

- Lazy package imports so `import mtplx` does not import MLX.
- No-MLX-safe `mtplx --help`, `doctor`, `inspect`, and `init` surface.
- Fresh-venv wheel smoke script for the Phase 0 install gate.
- Public benchmark dry-run paths that do not import heavy runtime modules.

### Known Caveats

- Sustained no-fan long-context throughput is below the 50+ tok/s target.
- `performance-cold` is opt-in and may require the MTPLX MLX fork.
- The curated release repository is private-first until QA passes.

### Roadmap

- v0.2: kernel ladder for sustained no-fan throughput.
- v0.3: additional MTP architectures and broader serving polish.
