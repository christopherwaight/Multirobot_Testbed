# Refactor clusterbuilder.py into a package

## Progress tracker

Update this section after each step. Use `[x]` for done, `[~]` for in-progress, `[ ]` for not started.

- [x] Step 1: Create package skeleton (`clusterbuilder/` dir, no `__init__.py` yet)
- [x] Step 2a: Extract `errors.py` (ClusterBuilderError)
- [x] Step 2b: Extract `backend.py` (NumpyBackend, SympyBackend)
- [x] Step 2c: Extract `tree.py` (TreeNode, TreeParser)
- [x] Step 2d: Extract `leaf_blocks.py` (PairBlock, TripleBlock)
- [x] Step 2e: Extract `hub_spoke.py` (HubAndSpokeKinematics)
- [x] Step 2f: Extract `coc.py` (ClusterOfClustersKinematics)
- [x] Step 2g: Extract `visualizer.py` (Visualizer)
- [x] Step 2h: Extract `codegen.py` (CodeGenerator)
- [x] Step 2i: Extract `cli.py` (parse_args, main)
- [x] Step 2j: Add `clusterbuilder/__init__.py` (atomic switchover; also exports `math` for Bug 6 monkey-patch compatibility)
- [x] Step 3: Replace `clusterbuilder.py` with launcher shim
- [x] Step 4: Snapshot tests pass without regeneration (14/14)
- [x] Step 5a: Add `tests/test_public_api.py`
- [x] Step 5b: Add `tests/test_codegen_units.py`
- [x] Step 5c: Add `tests/test_cli_smoke.py`
- [x] Final: 173 passing, 2 pre-existing Bug 6 failures; CLI smoke verified

Test baseline before refactor: 157/159 passing (2 failures are pre-existing Bug 6, unrelated).

Backup of the original `cluster_builder/` directory exists at
`trunk/Python Simulations/Vector_Fields/VF_Robot/cb2/` in case rollback is needed.

---

## Context

`trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder/clusterbuilder.py`
is a 2372-line monolith with 10 top-level classes covering tree parsing, two
kinematics families, two leaf blocks, the SAS/Pair runtime math, a sympy backend,
a matplotlib visualizer, and a 748-line code emitter for FK / IK / J_inv / J_fwd /
YAML / cluster wrapper. It is the only entry point: `tests/`, `yaml_visualize.py`,
and the CLI (`python clusterbuilder.py ...`) all import from it. The recent
battle-test session found 9 emitter bugs, all of which were hard to localize
because the emitter is one class with 17 mutually recursive methods, no module
boundary between FK / IK / Jacobian emission, and string templates interleaved
with the data they format.

The aim is to split it into a small package so that:

- Each math concept (tree, leaf blocks, hub-and-spoke kinematics, CoC kinematics,
  Jacobians) lives in its own file under 600 lines.
- The emitter is one file, separable from the runtime, so emitter bugs do not
  require reading runtime code to diagnose.
- The CLI invocation `python clusterbuilder.py ...` and every existing
  `from clusterbuilder import ...` line keeps working byte-for-byte.
- The existing 157/159-passing test suite continues to pass without edits.

This is verification-preserving refactor work, not a behavior change. The emitted
output must remain byte-identical (the `test_orientation_off_regression.py` and
`test_orientation_on_snapshot.py` snapshot tests will catch any drift).

## Approach

Convert `clusterbuilder.py` into a `clusterbuilder/` package directory. Move
`clusterbuilder.py` itself into a thin top-level launcher that calls
`clusterbuilder.cli.main()` so the CLI invocation pattern survives. The package
`__init__.py` re-exports every name currently imported from `clusterbuilder` by
tests or `yaml_visualize.py`, so no call site changes.

### Final layout

```
cluster_builder/
  clusterbuilder.py            # 5-line CLI launcher: from clusterbuilder.cli import main; main()
  clusterbuilder/
    __init__.py                # re-exports public API
    backend.py                 # _NumpyBackend, _SympyBackend, NumpyBackend, SympyBackend
    errors.py                  # ClusterBuilderError
    tree.py                    # TreeNode, TreeParser
    leaf_blocks.py             # PairBlock, TripleBlock
    hub_spoke.py               # HubAndSpokeKinematics
    coc.py                     # ClusterOfClustersKinematics
    visualizer.py              # Visualizer
    codegen.py                 # CodeGenerator (whole class, stays at ~750 lines)
    cli.py                     # parse_args, main
  tests/                       # unchanged
  yaml_visualize.py            # unchanged
  README.md                    # unchanged
```

Naming the package `clusterbuilder` (same as the existing module) means
`from clusterbuilder import HubAndSpokeKinematics` works via the package's
`__init__.py` exactly as today. Python resolves the package directory first when
both a `clusterbuilder/` directory and a `clusterbuilder.py` file exist in the
same parent, so the top-level `clusterbuilder.py` becomes a CLI-only launcher
(`if __name__ == "__main__": from clusterbuilder.cli import main; main()`).

### Public surface to re-export from `clusterbuilder/__init__.py`

```
from .backend     import SympyBackend, NumpyBackend
from .errors      import ClusterBuilderError
from .tree        import TreeNode, TreeParser
from .leaf_blocks import PairBlock, TripleBlock
from .hub_spoke   import HubAndSpokeKinematics
from .coc         import ClusterOfClustersKinematics
from .visualizer  import Visualizer
from .codegen     import CodeGenerator
```

`SympyBackend` is currently a module-level singleton instance (`SympyBackend =
_SympyBackend()`); preserve that instance form so `test_symbolic_vs_ik_diff.py`
keeps working.

### Splitting steps (in order)

Each step ends in a green test run. Do not move on until `pytest tests/ -v`
shows 157+ passing.

1. Create the package skeleton: `clusterbuilder/` directory and an empty
   `__init__.py`. Verify tests still pass against the old monolith.

2. Extract leaves, in dependency order:
   - `errors.py` (no deps)
   - `backend.py` (deps: numpy, optional sympy)
   - `tree.py` (deps: errors, dataclasses, ast)
   - `leaf_blocks.py` (deps: backend, math)
   - `hub_spoke.py` (deps: backend, math, numpy)
   - `coc.py` (deps: backend, tree, leaf_blocks, hub_spoke types, math, numpy)
   - `visualizer.py` (deps: matplotlib, hub_spoke, coc, tree)
   - `codegen.py` (deps: coc, hub_spoke, tree, visualizer, yaml)
   - `cli.py` (deps: tree, hub_spoke, coc, codegen, visualizer)

   After each extraction, add the re-export to `__init__.py` and run pytest.

3. Replace `clusterbuilder.py` with a launcher:
   ```python
   """clusterbuilder CLI launcher. Importable API lives in the clusterbuilder package."""
   if __name__ == "__main__":
       from clusterbuilder.cli import main
       main()
   ```

4. Snapshots should NOT need regenerating. If they do, the refactor broke
   something; the diff is the diagnostic.

### New tests to add

#### A. `tests/test_public_api.py` (~30 lines)

```python
def test_public_api_imports():
    from clusterbuilder import (
        HubAndSpokeKinematics, ClusterOfClustersKinematics,
        PairBlock, TripleBlock, TreeNode, TreeParser,
        Visualizer, CodeGenerator, ClusterBuilderError, SympyBackend,
    )
    for cls in (HubAndSpokeKinematics, ClusterOfClustersKinematics,
                PairBlock, TripleBlock, TreeNode, TreeParser,
                Visualizer, CodeGenerator):
        assert isinstance(cls, type), f"{cls.__name__} is not a class"
    assert issubclass(ClusterBuilderError, Exception)
    assert hasattr(SympyBackend, 'cos') and hasattr(SympyBackend, 'sin')
```

#### B. `tests/test_codegen_units.py` (~120 lines)

Unit tests for each `CodeGenerator.generate_*` method on a `(2,2,2)` CoC:
- `generate_yaml()` parses as YAML, contains `formation`, `config_tree`, `num_robots`.
- `generate_forward_kinematics()`, `generate_inverse_kinematics()`,
  `generate_inverse_jacobian()`, `generate_forward_jacobian()`,
  `generate_cluster_file()` each return a string that `compile()` accepts.

Guards against the f-string substitution bug class (Bug 3) without a full snapshot.

#### C. `tests/test_cli_smoke.py` (~40 lines)

Subprocess test for the CLI:

```python
def test_cli_image_only(tmp_path):
    cmd = [sys.executable, str(CLUSTERBUILDER_PY),
           "6", "2", "smoke", "image_only"]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "smoke_visualization.png").exists()

def test_cli_full(tmp_path):
    cmd = [sys.executable, str(CLUSTERBUILDER_PY),
           "6", "2", "smoke", "full", "(2,2,2)"]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for stem in ("smoke_visualization.png", "smoke.yaml",
                 "smoke_forward_kinematics.py", "smoke_inverse_kinematics.py",
                 "smoke_inverse_jacobian.py", "smoke_forward_jacobian.py",
                 "smoke_cluster.py"):
        assert (tmp_path / stem).exists(), f"missing: {stem}"
```

## Critical files

- `trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder/clusterbuilder.py`
- `trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder/tests/conftest.py`
- `trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder/yaml_visualize.py`
- `trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder/tests/snapshots/`

## Identifiers to preserve verbatim

- `SympyBackend` (instance) at `backend.py`.
- `PairBlock.j_inv`, `TripleBlock.j_inv` backend-pluggable signature.
- `TreeParser.parse`, `TreeParser.auto_build`, `TreeParser.validate`.
- `CodeGenerator.write_all(image_only=bool)`.
- All emitter method signatures inside `CodeGenerator`.
- CLI invocation: `python clusterbuilder.py num_robots cluster_config name run_mode [config_tree] [--symbolic] [--orientation on|off]`.

## Risks

- Circular imports between `visualizer`, `codegen`, kinematics modules. Use
  in-method or TYPE_CHECKING imports if needed.
- Snapshot drift on emitted output: treat any diff as a real regression.
- CLI resolution. From `cluster_builder/`, running `python clusterbuilder.py`
  may put the script's directory on `sys.path[0]` first. Inside the launcher,
  `from clusterbuilder.cli import main` must resolve to the package directory,
  not the script. Validate by running the smoke CLI command end-to-end.

## Verification

After each extraction step and at the end:

```bash
source "trunk/Python Simulations/Vector_Fields/VF_Robot/venv/bin/activate"
cd "trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder"
pytest tests/ -v
```

Pass criteria:
- 157+ tests pass.
- Snapshot tests pass without regeneration.
- `python clusterbuilder.py 6 2 smoke image_only` writes `smoke_visualization.png`.
- `python clusterbuilder.py 6 2 smoke full "(2,2,2)"` writes all seven files.
- `python yaml_visualize.py my_pentagon.yaml` runs without error.
- The three new tests pass.

## Out of scope

- No math changes, no emitted-output changes, no new features.
- No edits to the `experimental/cluster_builder/` duplicate.
- No deeper split of `CodeGenerator`.
- No `pyproject.toml` packaging.
