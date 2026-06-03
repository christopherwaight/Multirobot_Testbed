# clusterbuilder.py progress log

## Status: all tasks complete

All 37 tests pass: `python3 -m pytest tests/ -v`

---

## Checklist

- [x] 1a: phi -> psi rename in HTML (cluster_kinematics3.html Sections 0, 2, 9), quad_kinematics.py, quad_cluster.py, quad_*.yaml
- [x] 1b: Refactor hub-and-spoke inverse_jacobian to use col_idx dict (remove hand-indexed column arithmetic)
- [x] 1c: Snapshot orientation=off generated files for hexstar_snap (config 1) and my_star_snap (config 2)
- [x] 1d: FK/IK interfaces extended for orientation=on (3-tuple (x,y,theta) per robot)
- [x] 1e: theta_ref_var populated during _fk_node traversal (including size-1 leaf fix via _update_theta_ref_for_size1)
- [x] 1f: inverse_jacobian post-pass for heading rows; phi_i columns exactly zero in all position rows
- [x] 1g: ClusterOfClustersKinematics.inverse_jacobian accepts backend= parameter; _jac_node refactored
- [x] 1h: HubAndSpokeKinematics.inverse_jacobian accepts backend= parameter
- [x] 1i: Generated cluster.py: ORIENTATION_ENABLED flag, docstring note, RuntimeError guard in __init__
- [x] 1j: --orientation {on,off} CLI flag wired through parse_args and main()
- [x] Task 2a: NumpyBackend and SympyBackend classes added to clusterbuilder.py
- [x] Task 2b: PairBlock.j_inv and TripleBlock.j_inv accept backend= parameter
- [x] Task 2c: --symbolic CLI flag; generate_symbolic_jacobian() in CodeGenerator; write_all emits .txt and .tex
- [x] Task 2d: inverse_jacobian_symbolic() methods on both kinematics classes
- [x] Task 3a: Section 0 notation entries for psi and phi_i (done in 1a)
- [x] Task 3b: Big-O complexity subsection after "Recursive assembly for an arbitrary tree" (Section 6)
- [x] Task 3c: atom-library math-note in Section 8
- [x] Task 4a: config_tree field added to generate_yaml for cluster-of-clusters; TreeParser.to_string() added
- [x] Task 4b: Visualizer.render_hub_spoke and render_cluster_of_clusters accept state=None parameter
- [x] Task 4c: yaml_visualize.py written; reads YAML, dispatches by type, renders via Visualizer
- [x] Task 5: tests/test_fd_jacobian.py (14 tests: hub-spoke N=4,5 and 5 CoC trees, both orientation settings)
- [x] Task 5: tests/test_orientation_off_regression.py (8 tests: byte-for-byte match with snapshots)
- [x] Task 6: tests/test_symbolic_jacobian.py (7 tests: numeric match at default state, identity check)
- [x] Task 7: tests/test_yaml_roundtrip.py (8 tests: smoke, backwards compat, edited-state-reaches-image)
- [x] Task 8: README.md updated with --orientation, --symbolic, config_tree, yaml_visualize.py
- [x] Task 9: clusterbuilder_progress.md (this file)

---

## Key design decisions

- **theta_ref_var for size-1 leaves**: size-1 leaf nodes don't own a theta variable; their parent's
  theta_key is not known until the parent finishes. Solution: pass parent_theta_key down into _fk_node,
  and add _update_theta_ref_for_size1 to retroactively update size-1 leaf children after any internal
  node discovers its own theta key.

- **Absolute angle convention**: child theta variables are global, not relative to parent. Only
  columns 0:2 of S_c (position sensitivity) chain through the parent Jacobian. Column 2 (theta
  sensitivity) is never triggered by a parent variable change.

- **Backend abstraction**: NumpyBackend and SympyBackend share the same interface (cos, sin, zeros,
  array). The same inverse_jacobian recursion runs with either backend, guaranteeing the symbolic
  output matches the numeric output by construction.

- **_chain2 helper**: element-by-element dot product (not @ operator) to avoid ndim>0 deprecation
  warnings from numpy and to work natively with sympy expressions.

- **Orientation=on cluster.py guard**: generated file raises RuntimeError rather than emitting a
  silently-wrong 2-DOF controller. ORIENTATION_ENABLED = True class attribute and module docstring
  make the file self-describing.

- **config_tree YAML field**: TreeParser.to_string() serializes any TreeNode back to a parseable
  string. Added to cluster-of-clusters generate_yaml output only; hub-and-spoke omits it.
  Backwards-compatible: existing consumers never read config_tree.
