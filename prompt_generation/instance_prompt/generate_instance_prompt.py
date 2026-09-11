"""
Generate instance-level prompts from 3D instance segmentation results.

Input:
    A PLY file containing at least:
        - semantic: semantic class ID
        - instance: instance ID

Output:
    A text prompt describing the number of detected object instances
    for each semantic category.

Example:
    This indoor scene contains 13 Chairs, 1 Table, 2 Sofas.
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from plyfile import PlyData


# ----------------------------------------------------------------------
# Semantic label mapping
# Modify this mapping according to the dataset used in your experiments.
# ----------------------------------------------------------------------
DEFAULT_LABEL_MAP = {
    0: "Ceiling",
    1: "Floor",
    2: "Wall",
    3: "Beam",
    4: "Column",
    5: "Window",
    6: "Door",
    7: "Table",
    8: "Chair",
    9: "Sofa",
    10: "Bookcase",
    11: "Board",
}


# ----------------------------------------------------------------------
# Classes that are treated as structural elements rather than movable
# objects. Adjust this list according to your application.
# ----------------------------------------------------------------------
DEFAULT_STRUCTURAL_CLASSES = {
    "Ceiling",
    "Floor",
    "Wall",
    "Beam",
    "Column",
    "Window",
    "Door",
}


def read_ply_labels(
    ply_path,
    semantic_field="semantic",
    instance_field="instance"
):
    """
    Read semantic and instance labels from a PLY file.

    Parameters
    ----------
    ply_path : str or Path
        Input PLY file.
    semantic_field : str
        Name of the semantic label field.
    instance_field : str
        Name of the instance label field.

    Returns
    -------
    semantic_labels : np.ndarray
    instance_labels : np.ndarray
    """

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data

    if semantic_field not in vertex.dtype.names:
        raise ValueError(
            f"Semantic field '{semantic_field}' not found in {ply_path}. "
            f"Available fields: {vertex.dtype.names}"
        )

    if instance_field not in vertex.dtype.names:
        raise ValueError(
            f"Instance field '{instance_field}' not found in {ply_path}. "
            f"Available fields: {vertex.dtype.names}"
        )

    semantic_labels = np.asarray(vertex[semantic_field])
    instance_labels = np.asarray(vertex[instance_field])

    return semantic_labels, instance_labels


def build_instance_statistics(
    semantic_labels,
    instance_labels,
    label_map,
    ignore_instance_ids=None,
    exclude_structural=True,
):
    """
    Build instance statistics from semantic and instance labels.

    Each unique instance ID is associated with one semantic category.
    The number of instances belonging to each category is then counted.

    Returns
    -------
    category_counts : dict
        Example:
            {
                "Chair": 13,
                "Table": 1,
                "Sofa": 2
            }
    """

    if ignore_instance_ids is None:
        ignore_instance_ids = {0, -1}

    instance_to_semantic = {}

    unique_instances = np.unique(instance_labels)

    for instance_id in unique_instances:

        if instance_id in ignore_instance_ids:
            continue

        mask = instance_labels == instance_id

        if not np.any(mask):
            continue

        semantic_values = semantic_labels[mask]

        # Use the most frequent semantic label within the instance.
        unique_labels, counts = np.unique(
            semantic_values,
            return_counts=True
        )

        semantic_id = int(unique_labels[np.argmax(counts)])

        category_name = label_map.get(
            semantic_id,
            f"Class_{semantic_id}"
        )

        if exclude_structural and category_name in DEFAULT_STRUCTURAL_CLASSES:
            continue

        instance_to_semantic[instance_id] = category_name

    # Count instances for each semantic category.
    category_counts = defaultdict(int)

    for category_name in instance_to_semantic.values():
        category_counts[category_name] += 1

    return dict(category_counts)


def pluralize_category(category, count):
    """
    Convert a category name into a simple plural form.

    Examples:
        Chair -> Chairs
        Sofa  -> Sofas
        Bookcase -> Bookcases

    For irregular nouns, explicit rules can be added here.
    """

    if count == 1:
        return category

    irregular = {
        "Person": "People",
        "Child": "Children",
        "Mouse": "Mice",
    }

    if category in irregular:
        return irregular[category]

    if category.endswith("s"):
        return category

    if category.endswith("x"):
        return category + "es"

    if category.endswith("ch"):
        return category + "es"

    if category.endswith("sh"):
        return category + "es"

    return category + "s"


def generate_instance_prompt(category_counts):
    """
    Generate the final instance-level prompt.

    Example:
        This indoor scene contains 13 Chairs, 1 Table, 2 Sofas.
    """

    if not category_counts:
        return "This indoor scene contains no identifiable object instances."

    # Sort by semantic class ID-independent alphabetical order.
    # This provides deterministic output.
    sorted_items = sorted(
        category_counts.items(),
        key=lambda x: x[0]
    )

    object_descriptions = []

    for category, count in sorted_items:

        category_text = pluralize_category(
            category,
            count
        )

        object_descriptions.append(
            f"{count} {category_text}"
        )

    return (
        "This indoor scene contains "
        + ", ".join(object_descriptions)
        + "."
    )


def process_single_file(
    input_path,
    output_path,
    label_map,
    semantic_field="semantic",
    instance_field="instance",
    exclude_structural=True,
):
    """
    Process one instance-segmentation PLY file.
    """

    semantic_labels, instance_labels = read_ply_labels(
        input_path,
        semantic_field=semantic_field,
        instance_field=instance_field,
    )

    category_counts = build_instance_statistics(
        semantic_labels,
        instance_labels,
        label_map,
        exclude_structural=exclude_structural,
    )

    prompt = generate_instance_prompt(category_counts)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt + "\n")

    return prompt


def process_directory(
    input_dir,
    output_dir,
    label_map,
    semantic_field="semantic",
    instance_field="instance",
    exclude_structural=True,
):
    """
    Process all PLY files in a directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ply_files = sorted(input_dir.glob("*.ply"))

    if not ply_files:
        print(f"No PLY files found in {input_dir}")
        return

    for ply_path in ply_files:

        output_path = (
            output_dir /
            f"{ply_path.stem}_instance.txt"
        )

        prompt = process_single_file(
            ply_path,
            output_path,
            label_map,
            semantic_field=semantic_field,
            instance_field=instance_field,
            exclude_structural=exclude_structural,
        )

        print(f"[OK] {ply_path.name}")
        print(f"     {prompt}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate instance-level prompts from "
            "3D instance segmentation results."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input PLY file or directory."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for generated prompts."
    )

    parser.add_argument(
        "--semantic_field",
        default="semantic",
        help="Semantic label field name in the PLY."
    )

    parser.add_argument(
        "--instance_field",
        default="instance",
        help="Instance label field name in the PLY."
    )

    parser.add_argument(
        "--include_structural",
        action="store_true",
        help=(
            "Include structural elements such as walls, floors, "
            "ceilings, etc. in the instance prompt."
        )
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    exclude_structural = not args.include_structural

    if input_path.is_file():

        output_path = (
            output_dir /
            f"{input_path.stem}_instance.txt"
        )

        prompt = process_single_file(
            input_path,
            output_path,
            DEFAULT_LABEL_MAP,
            semantic_field=args.semantic_field,
            instance_field=args.instance_field,
            exclude_structural=exclude_structural,
        )

        print(prompt)

    elif input_path.is_dir():

        process_directory(
            input_path,
            output_dir,
            DEFAULT_LABEL_MAP,
            semantic_field=args.semantic_field,
            instance_field=args.instance_field,
            exclude_structural=exclude_structural,
        )

    else:
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )


if __name__ == "__main__":
    main()
