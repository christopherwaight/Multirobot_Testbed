# Cluster Builder

Automatically generates kinematics, Jacobians, a config file, a simulation class, and a visualization for any N-robot formation.

---

## How to Run

Step 1, Change directory to:  /Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder

Step 2, run ../venv/bin/python3 clusterbuilder.py <num_robots> <cluster_config> <cluster_name> <run_mode> [config_tree]

Examples to show ckitts:


../venv/bin/python3 clusterbuilder.py 4 2 my_diamond_with_phi full --symbolic --orientation on
../venv/bin/python3 clusterbuilder.py 4 2 my_diamond full --symbolic

../venv/bin/python3 clusterbuilder.py 6 2 my_star image_only
../venv/bin/python3 clusterbuilder.py 6 2 my_star2 image_only "(3,3)"
../venv/bin/python3 yaml_visualize.py my_pentagon.yaml

../venv/bin/python3 clusterbuilder.py 6 1 my_star image_only 


```bash
cd "/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python Simulations/Vector_Fields/VF_Robot/cluster_builder"
../venv/bin/python3 clusterbuilder.py <num_robots> <cluster_config> <cluster_name> <run_mode> [config_tree]
```

| Argument | What it means |
|---|---|
| `num_robots` | How many robots in the formation |
| `cluster_config` | `1` = hub-and-spoke &nbsp;&nbsp; `2` = cluster-of-clusters |
| `cluster_name` | Name for your output files (no spaces) |
| `run_mode` | `image_only` = just draw it &nbsp;&nbsp; `full` = generate all 7 files |
| `config_tree` | *(Optional, config 2 only)* Custom tree structure — see below |

---

## Three Examples

### Example 1 — 5-robot star, just see the picture

```bash
../venv/bin/python3 clusterbuilder.py 5 1 my_star image_only
```

One hub robot in the center, four spokes around it. Outputs `my_star_visualization.png` so you can check the layout before generating any code.

---

### Example 2 — 6-robot formation, three pairs in a triangle, full output

```bash
../venv/bin/python3 clusterbuilder.py 6 2 hex_formation full "(2,2,2)"
```

Three pairs of robots, where the three pair-centroids form a triangle. The `(2,2,2)` tells it to split 6 robots into three groups of 2. Generates all 7 files:

```
hex_formation_forward_kinematics.py   ← forward kinematics function
hex_formation_inverse_kinematics.py   ← inverse kinematics function
hex_formation_forward_jacobian.py     ← forward Jacobian
hex_formation_inverse_jacobian.py     ← inverse Jacobian
hex_formation.yaml                    ← formation config (load into your cluster class)
hex_formation_cluster.py              ← ready-to-use cluster class
hex_formation_visualization.png       ← diagram of the formation
```

---

### Example 3 — 15-robot formation, let it figure out the structure automatically

```bash
../venv/bin/python3 clusterbuilder.py 15 2 big_cluster full
```

No `config_tree` needed — the tool automatically splits 15 robots into a balanced tree of sub-clusters and generates all 7 files. Good starting point when you just want something that works.

---

## Custom Tree Structure (Optional)

For cluster-of-clusters (config 2), you can specify exactly how robots are grouped using a nested tuple. Leaves must be `1`, `2`, or `3`. Each node can have **2 or 3 children**.

```
(2, 2)           →  4 robots: two pairs
(2, 2, 2)        →  6 robots: three pairs in a triangle
(3, 3)           →  6 robots: two triangles
((2,2), (2,2))   →  8 robots: two groups of two pairs
((2,3), (2,3))   → 10 robots: two groups of (pair + triangle)
```

Example — 9 robots as three triangles arranged in a larger triangle:

```bash
../venv/bin/python3 clusterbuilder.py 9 2 tri_of_tris full "(3,3,3)"
```

---

## Optional flags

### `--orientation {on,off}` (default off)

Adds a per-robot heading `phi_i` to the state, growing it from `2N` to `3N` variables. Each robot gains an angle that spins it in place without moving its position. The relationship is:

    theta_i = theta_ref(i) + phi_i

where `theta_ref(i)` is the orientation of the immediate parent frame robot `i` belongs to. In hub-and-spoke this is always `theta_c`; in cluster-of-clusters it is the `theta_{nid}` of the leaf that contains robot `i`.

The forward Jacobian, inverse Jacobian, FK, and IK are all fully correct when orientation is on. The generated `*_cluster.py` includes a `RuntimeError` guard with a clear message rather than emitting a silently-wrong 2-DOF controller.

```bash
../venv/bin/python3 clusterbuilder.py 5 1 pentagon full --orientation on
```

### `--symbolic` (default off)

Emits two additional files alongside the normal output:

- `<name>_inverse_jacobian_symbolic.txt` -- pretty-printed symbolic matrix via sympy
- `<name>_inverse_jacobian_symbolic.tex` -- LaTeX representation for inclusion in documents

Requires `sympy`. The symbolic Jacobian uses the same recursive path as the numeric one (same code, different backend), so it is guaranteed to match.

```bash
../venv/bin/python3 clusterbuilder.py 4 2 hframe full "(2,2)" --symbolic
```

---

## Visualizing a YAML file

`yaml_visualize.py` renders a formation PNG from any clusterbuilder-generated YAML:

```bash
../venv/bin/python3 yaml_visualize.py hframe.yaml
../venv/bin/python3 yaml_visualize.py hframe.yaml --out my_output.png
```

For cluster-of-clusters YAMLs, the `config_tree` field (written automatically by `clusterbuilder.py full` mode since Task 4) enables exact round-trip rendering. If the field is absent, the renderer falls back to `TreeParser.auto_build(N)` with a warning -- this only matches the original layout for auto-built trees.

---

## Notes

- All output files are written to the current directory (`cluster_builder/`)
- The generated `*_cluster.py` file follows the same pattern as `OmniCluster` and `QuadCluster` -- plug it straight into your existing simulation runner
- `beta = 0` (collinear robots) is a singularity -- the tool's defaults are always set to equilateral triangles (60 degrees) to stay safely away from it
- Hub-and-spoke (config 1) does not use a `config_tree` -- it is always one hub plus N-1 spokes
- With `--orientation on`, the generated cluster file raises a `RuntimeError` to prevent accidental use as a simulation controller. The kinematics and Jacobians are mathematically complete and verified by the FD test suite.
