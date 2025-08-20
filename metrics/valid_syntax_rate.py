import sys
import os
import cadquery as cq
import numpy as np
import textwrap
from typing import Union, Dict, List

os.environ["CADQUERY_LOG_LEVEL"] = "ERROR"


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


def evaluate_syntax_rate(
    codes: Dict[str, str], verbose: bool = True
) -> Dict[str, Union[float, int, List[str]]]:
    """Evaluate valid syntax rate for a dictionary of CadQuery code strings.

    Args:
        codes: Dict with IDs as keys and Python code strings as values
        verbose: Whether to print detailed results

    Returns:
        Dict with 'vsr' (valid syntax rate), 'successful' (count), 'total' (count),
        'failed_ids' (list of IDs that failed)
    """
    if not codes:
        if verbose:
            print("No code provided")
        return {"vsr": 0.0, "successful": 0, "total": 0, "failed_ids": []}

    ids = sorted(codes.keys())
    successful_count = 0
    failed_ids = []

    for script_id in ids:
        code = codes[script_id]
        try:
            solid = _load_solid_from_code(code, script_id)
            successful_count += 1
            if verbose:
                print(f"✓ {script_id}: Successfully executed")
        except Exception as exc:
            failed_ids.append(script_id)
            if verbose:
                print(f"✗ {script_id}: {exc}")

    total_count = len(ids)
    vsr = successful_count / total_count if total_count > 0 else 0.0

    if verbose:
        print(f"\n--- SUMMARY ---")
        print(f"Successful: {successful_count}/{total_count}")
        print(f"Valid Syntax Rate: {vsr:.3f}")
        if failed_ids:
            print(f"Failed IDs: {failed_ids}")

    return {
        "vsr": vsr,
        "successful": successful_count,
        "total": total_count,
        "failed_ids": failed_ids,
    }


def evaluate_syntax_rate_simple(codes: Dict[str, str]) -> float:
    """Simple function that just returns the valid syntax rate as a float."""
    result = evaluate_syntax_rate(codes, verbose=False)
    return result["vsr"]


if __name__ == "__main__":
    # Test cases
    test_codes = {
        "simple_box": """
            height = 60.0
            width = 80.0
            thickness = 10.0
            result = cq.Workplane("XY").box(height, width, thickness)
        """,
        "box_with_hole": """
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
        """,
        "syntax_error": """
            result = cq.Workplane("XY").box(10, 10, 10
            # Missing closing parenthesis
        """,
        "runtime_error": """
            result = cq.Workplane("XY").box(undefined_variable, 10, 10)
        """,
        "no_cadquery_object": """
            x = 5
            y = 10
            z = x + y
        """,
    }

    print("Testing Valid Syntax Rate evaluation:")
    print("=" * 50)

    result = evaluate_syntax_rate(test_codes)
    print(f"\nOverall VSR: {result['vsr']:.1%}")
