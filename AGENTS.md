# AGENTS.md

TinyMachines.jl is a single Julia package (no monorepo) implementing small Flux models for semantic segmentation (UNet4/5, MobileUNet, ESPNet + PReLU). All models output logits only (final activations removed in v0.3.7; apply `sigmoid` etc. yourself). Feature maps are available via `Flux.activations` on sub-chains.

## Setup

- Always work with the local project activated: `julia --project` (or `julia --project=.` from anywhere).
- After fresh clone or dep change: `julia --project -e 'using Pkg; Pkg.instantiate()'`.
- `Manifest.toml` is listed in `.gitignore` (never commit it).
- Julia 1.10–1.13 + Flux 0.14–0.16 (enforced in `Project.toml`).

## Testing

- Full test suite: `julia --project -e 'using Pkg; Pkg.test()'` or REPL `] test`.
- Entry point: `test/runtests.jl` (sets `Random.seed!(1234)`, defines 256×256 test inputs `x3`/`x4`/`x8`, then includes the other `*-tests.jl` files under one `@testset`).
- `test/epoch-tests.jl` (uses `Flux.setup` + `Flux.train!` + `logitcrossentropy(..., dims=3)`) is commented out in `runtests.jl`.
- Focused runs: edit the `include` list in `runtests.jl` temporarily, then run the command above. (No test filtering or `--quick` support.)
- Tests are CPU-only. Many verify output sizes and `Flux.activations(model.encoder, x)` (or `.d` for MobileUNet).

## Development commands

- Edit source under `src/`, then `] test` (Pkg reloads the package automatically).
- Add/remove deps: `] add Foo`, `] rm Foo` (edit `Project.toml`; do not touch Manifest).
- No formatters, linters, typecheckers, codegen, or pre-commit hooks are configured.
- Example usage/training patterns live in `examples/train.ipynb` and the epoch test (use `Flux.DataLoader`, `Flux.setup`, `Flux.train!`).

## Source layout & entrypoints

- `src/TinyMachines.jl`: module definition, imports from Flux, `include`s, exports (`UNet*`, `MobileUNet`, `ESPNet`, lowercase ctors, `PReLU`).
- `src/functions/`: reusable building blocks (not re-exported):
  - `misc.jl` (internal `fpos`/`fneg`)
  - `convlayers.jl` (ConvK1/K2/K3, ConvTrK*, DilatedConvK3 + aliases)
  - `activationlayers.jl` (PReLU struct + call)
  - `unetblocks.jl` (CB, MCB helpers)
  - `irblocks.jl`
  - `espmodules.jl` (defines `DownSampler` and `ESPModule`)
- `src/models/`: model structs + constructors:
  - `unet5.jl` (also `UNet`, `unet`, `unet5`)
  - `unet4.jl`
  - `mobileunet.jl`
  - `espnet.jl` (defines `espnet`/`ESPNet` split into 3 encoder stages + dropout-on-main-path only; `espnet_original.jl` kept for reference)
- Models are callable structs with `@layer`. Forward passes often do manual `.layers.xxx` indexing (e.g. `m.encoder.layers.e1(x)`) rather than treating sub-chains as full functions.
- ESPNet quirk: `activation="prelu"` (string) or a Function. PReLU resolution happens inside the constructors before calling the conv helpers. After the stage split, image fusion uses explicit `cat` (plus threading of original image) inside the stage call methods (`ESPEncoderStage*`); `Parallel` is not used for ESPNet. Encoder activations return `(fused, x)` tuples for stages 1/2 (to support image ds) and the fused array for stage 3.

## Other notes

- Convenience ctors (`UNet5()`, `ESPNet()`) hard-code the "recommended" hyperparams from the paper implementations.
- Full ctors (`unet5(...)`, `espnet(...)`) expose `alpha`, dropouts, etc.
- No return-features API remains (removed v0.5.0; use `Flux.activations`).
- CI (`.github/workflows/CI.yml`) runs the full suite via `julia-runtest` on the matrix in Project.toml; other workflows are stock (TagBot, CompatHelper, dependabot).

See `README.md` (model signatures), `CHANGELOG.md`, and `Project.toml` for the executable source of truth. When in doubt, run the commands above and inspect the actual `include`d files rather than prose.