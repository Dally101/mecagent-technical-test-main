import argparse, importlib.util, runpy, tempfile, itertools, sys
from pathlib import Path
import os
import cadquery as cq
from cadquery import exporters
import numpy as np
import trimesh
from typing import Union
import textwrap

os.environ["CADQUERY_LOG_LEVEL"] = "ERROR"


# ---------- helpers ---------------------------------------------------------


def _load_solid_from_code(
    code: str, script_id: str = "unknown"
) -> Union[cq.Solid, cq.Compound]:
    """Execute Python code and return any CadQuery object found."""
    # Clean up indentation issues
    cleaned_code = textwrap.dedent(code).strip()

    # Provide necessary imports in the execution namespace
    ns = {"cq": cq, "cadquery": cq, "np": np, "numpy": np, "__builtins__": __builtins__}
    try:
        exec(cleaned_code, ns)
    except Exception as e:
        raise ValueError(f"Error executing script {script_id}: {e}")

    # Find any CadQuery objects in the namespace
    cadquery_objects = []
    for var_name, var_value in ns.items():
        if isinstance(var_value, (cq.Workplane, cq.Solid, cq.Compound)):
            cadquery_objects.append((var_name, var_value))

    if not cadquery_objects:
        raise ValueError(
            f"No CadQuery objects (Workplane, Solid, or Compound) found in script {script_id}"
        )

    if len(cadquery_objects) > 1:
        # If multiple objects, prefer common names
        preferred_names = ["solid", "result", "shape", "part", "object", "obj", "res"]
        for preferred in preferred_names:
            for var_name, var_value in cadquery_objects:
                if var_name == preferred:
                    cadquery_objects = [(var_name, var_value)]
                    break
            if len(cadquery_objects) == 1:
                break

        # If still multiple, just take the first one but warn
        if len(cadquery_objects) > 1:
            var_names = [name for name, _ in cadquery_objects]
            print(
                f"Warning: Multiple CadQuery objects found in {script_id}: {var_names}. Using '{cadquery_objects[0][0]}'"
            )

    var_name, solid_obj = cadquery_objects[0]
    # print(
    #     f"Found CadQuery object in variable '{var_name}' of type {type(solid_obj).__name__}"
    # )

    # Handle different CadQuery object types
    if isinstance(solid_obj, cq.Workplane):
        # Extract the solid from the workplane
        solid_obj = solid_obj.val()

    # Handle Compound objects (multiple solids combined)
    if hasattr(solid_obj, "Solids") and callable(getattr(solid_obj, "Solids")):
        solids = solid_obj.Solids()
        if len(solids) == 1:
            solid_obj = solids[0]
        elif len(solids) > 1:
            # If multiple solids, we need to combine them into one
            # Use the compound itself if it's valid for our purposes
            pass  # Keep the compound as is
        else:
            raise ValueError(f"No solids found in compound in script {script_id}")

    # Accept both Solid and Compound objects for our mesh operations
    if not isinstance(solid_obj, (cq.Solid, cq.Compound)):
        raise ValueError(
            f"CadQuery object '{var_name}' is not a Solid or Compound object in script {script_id}, got {type(solid_obj)}"
        )

    return solid_obj


def _load_solid(script_path: Path) -> cq.Solid:
    """Import a CadQuery script in isolation and return the 'solid' object."""
    ns = runpy.run_path(script_path)  # executes the file
    if "solid" not in ns or not isinstance(ns["solid"], cq.Solid):
        raise ValueError(f"'solid' not found in {script_path}")
    return ns["solid"]


def _root_gyration(solid: Union[cq.Solid, cq.Compound]) -> float:
    vol = solid.Volume()
    inertia = np.array(cq.Shape.matrixOfInertia(solid)).reshape(3, 3)
    return np.sqrt(np.trace(inertia) / (2.0 * vol))


def _normalized_mesh(
    solid: Union[cq.Solid, cq.Compound], pitch: float = 0.01
) -> trimesh.Trimesh:
    """Translate to centroid, isotropically scale by r_g, and return a mesh."""
    r_g = _root_gyration(solid)
    center_vector = solid.Center()
    centroid = np.array([center_vector.x, center_vector.y, center_vector.z])
    # Export to temporary STL then load with trimesh
    with tempfile.TemporaryDirectory() as tmp:
        stl_path = Path(tmp) / "part.stl"
        exporters.export(solid, str(stl_path))
        mesh = trimesh.load(str(stl_path), force="mesh")
    mesh.apply_translation(-centroid)
    mesh.apply_scale(1.0 / r_g)
    return mesh


def _principal_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return 3×3 orthonormal matrix whose columns are principal axes."""
    inertia = mesh.moment_inertia
    _, vecs = np.linalg.eigh(inertia)
    return vecs  # columns are eigenvectors


def _apply_rotation(mesh: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    T = np.eye(4)
    T[:3, :3] = R
    mesh_rot = mesh.copy()
    mesh_rot.apply_transform(T)
    return mesh_rot


def _voxel_bool_unified(
    mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, pitch: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Create voxel grids for both meshes using unified bounds."""
    # Voxelize each mesh individually first
    voxel1 = mesh1.voxelized(pitch)
    voxel2 = mesh2.voxelized(pitch)

    # Get the bounds of each voxel grid
    bounds1 = voxel1.bounds
    bounds2 = voxel2.bounds

    # Compute unified bounds
    min_bounds = np.minimum(bounds1[0], bounds2[0])
    max_bounds = np.maximum(bounds1[1], bounds2[1])

    # Calculate grid dimensions
    grid_size = np.ceil((max_bounds - min_bounds) / pitch).astype(int)

    # Create empty unified voxel grids
    vox1 = np.zeros(grid_size, dtype=bool)
    vox2 = np.zeros(grid_size, dtype=bool)

    # Calculate offsets for placing each voxel grid in the unified space
    offset1 = np.round((bounds1[0] - min_bounds) / pitch).astype(int)
    offset2 = np.round((bounds2[0] - min_bounds) / pitch).astype(int)

    # Get shapes of individual voxel matrices
    shape1 = voxel1.matrix.shape
    shape2 = voxel2.matrix.shape

    # Calculate end positions
    end1 = offset1 + shape1
    end2 = offset2 + shape2

    # Place voxels in unified grids with bounds checking
    if np.all(offset1 >= 0) and np.all(end1 <= grid_size):
        vox1[offset1[0] : end1[0], offset1[1] : end1[1], offset1[2] : end1[2]] = (
            voxel1.matrix
        )

    if np.all(offset2 >= 0) and np.all(end2 <= grid_size):
        vox2[offset2[0] : end2[0], offset2[1] : end2[1], offset2[2] : end2[2]] = (
            voxel2.matrix
        )

    return vox1, vox2


def _voxel_bool(mesh: trimesh.Trimesh, pitch: float = 0.05) -> np.ndarray:
    vox = mesh.voxelized(pitch)
    return vox.matrix  # boolean 3-D numpy array


def iou_best(
    mesh_gt: trimesh.Trimesh, mesh_pred: trimesh.Trimesh, pitch: float = 0.05
) -> float:
    """IOU after best principal-axis alignment (4 valid sign flips)."""
    axes_gt = _principal_axes(mesh_gt)
    axes_pr = _principal_axes(mesh_pred)

    best = 0.0
    for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]:
        D = np.diag(signs)
        axes_pr_flipped = axes_pr @ D  # change axis directions
        R = axes_gt @ axes_pr_flipped.T  # rotation to align
        m_aligned = _apply_rotation(mesh_pred, R)

        # Use unified voxelization
        vox_gt, vox_pr = _voxel_bool_unified(mesh_gt, m_aligned, pitch)

        inter = np.logical_and(vox_gt, vox_pr).sum()
        union = np.logical_or(vox_gt, vox_pr).sum()

        if union > 0:
            iou = inter / union
            best = max(best, iou)

    return best


# ---------- main ------------------------------------------------------------


def evaluate_codes(gt_codes: dict, pred_codes: dict, pitch: float = 0.05):
    """Evaluate predictions against ground-truth using Python code directly.

    Args:
        gt_codes: Dict with IDs as keys and ground-truth Python code as values
        pred_codes: Dict with IDs as keys and prediction Python code as values
        pitch: Voxel pitch for IoU calculation
    """
    ids = sorted(gt_codes.keys())
    if not ids:
        sys.exit("no ground-truth scripts provided")

    vsr_success = 0
    ious = []

    for _id in ids:
        if _id not in pred_codes:
            print(f"missing prediction for {_id}, skipping")
            continue

        try:
            solid_gt = _load_solid_from_code(gt_codes[_id], f"gt_{_id}")
            solid_pr = _load_solid_from_code(pred_codes[_id], f"pred_{_id}")
            vsr_success += 1
        except Exception as exc:
            print(f"{_id}: syntax/runtime error -> {exc}")
            continue

        mesh_gt = _normalized_mesh(solid_gt)
        mesh_pr = _normalized_mesh(solid_pr)
        ious.append(iou_best(mesh_gt, mesh_pr, pitch))

    n_total = len(ids)
    vsr = vsr_success / n_total if n_total else 0.0
    iou_b = np.mean(ious) if ious else 0.0

    print(f"Valid Syntax Rate: {vsr:.3f}")
    print(f"Mean IOU_best   : {iou_b:.3f}")

    return {"vsr": vsr, "iou_best": iou_b}


def evaluate(gt_dir: Path, pred_dir: Path, pitch: float = 0.05):
    """Original file-based evaluation function."""
    ids = sorted(p.stem for p in gt_dir.glob("*.py"))
    if not ids:
        sys.exit("no ground-truth scripts found")

    vsr_success = 0
    ious = []

    for _id in ids:
        gt_path = gt_dir / f"{_id}.py"
        pr_path = pred_dir / f"{_id}.py"
        if not pr_path.exists():
            print(f"missing prediction for {_id}, skipping")
            continue

        try:
            solid_gt = _load_solid(gt_path)
            solid_pr = _load_solid(pr_path)
            vsr_success += 1
        except Exception as exc:
            print(f"{_id}: syntax/runtime error -> {exc}")
            continue

        mesh_gt = _normalized_mesh(solid_gt)
        mesh_pr = _normalized_mesh(solid_pr)
        ious.append(iou_best(mesh_gt, mesh_pr, pitch))

    n_total = len(ids)
    vsr = vsr_success / n_total if n_total else 0.0
    iou_b = np.mean(ious) if ious else 0.0

    print(f"Valid Syntax Rate: {vsr:.3f}")
    print(f"Mean IOU_best   : {iou_b:.3f}")


def get_iou_best(code1: str, code2: str):
    solid1 = _load_solid_from_code(code1)
    solid2 = _load_solid_from_code(code2)
    mesh1 = _normalized_mesh(solid1)
    mesh2 = _normalized_mesh(solid2)
    iou = iou_best(mesh1, mesh2)
    return iou


if __name__ == "__main__":
    code1 = """
        height = 60.0
        width = 80.0
        thickness = 10.0
        res = cq.Workplane("XY").box(height, width, thickness)
    """
    code2 = """
        height = 60.0
 width = 80.0
 thickness = 10.0
 diameter = 22.0
 padding = 12.0

 # make the base
 result = (
     cq.Workplane("XY")
     .box(height, width, thickness)
     .faces(">Z")
     .workplane()
     .hole(diameter)
     .faces(">Z")
     .workplane()
     .rect(height - padding, width - padding, forConstruction=True)
     .vertices()
     .cboreHole(2.4, 4.4, 2.1)
 )
    """
    solid1 = _load_solid_from_code(code1)
    solid2 = _load_solid_from_code(code2)
    mesh1 = _normalized_mesh(solid1)
    mesh2 = _normalized_mesh(solid2)
    iou = iou_best(mesh1, mesh2)
    print(f"IOU: {iou}")
