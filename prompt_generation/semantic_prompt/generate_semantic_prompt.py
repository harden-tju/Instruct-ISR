```python
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData


# ------------------------------------------------------------
# Default indoor scene categories
# Modify this mapping according to the label definition
# of the dataset being processed.
# ------------------------------------------------------------
DEFAULT_LABEL_MAP = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "table",
    8: "chair",
    9: "sofa",
    10: "bookcase",
    11: "board",
}


def load_semantic_labels(filename, label_field="semantic"):
    """
    Load semantic labels from a PLY file.

    Parameters
    ----------
    filename : str or Path
        Input PLY file.
    label_field : str
        Name of the semantic-label property in the PLY file.

    Returns
    -------
    labels : ndarray
        Semantic label for each point.
    """

    ply = PlyData.read(filename)
    vertex = ply["vertex"].data

    available_fields = vertex.dtype.names

    if label_field not in available_fields:
        raise ValueError(
            f"Label field '{label_field}' was not found in {filename}.\n"
            f"Available fields: {available_fields}"
        )

    return np.asarray(vertex[label_field])


def generate_semantic_prompt(
    labels,
    label_map,
    sort_by_label=True
):
    """
    Generate a semantic-level prompt from point-wise semantic labels.

    The resulting prompt follows the format used in the manuscript:

        This indoor scene contains Chair, Table, Sofa, ...

    Parameters
    ----------
    labels : ndarray
        Point-wise semantic labels.
    label_map : dict
        Mapping from label IDs to semantic category names.
    sort_by_label : bool
        Whether to keep categories in label-ID order.

    Returns
    -------
    prompt : str
        Semantic-level prompt.
    categories : list
        Categories present in the scene.
    """

    # Obtain unique semantic labels
    unique_labels = np.unique(labels)

    # Keep only labels defined in the label map
    valid_labels = [
        int(label)
        for label in unique_labels
        if int(label) in label_map
    ]

    if sort_by_label:
        valid_labels.sort()

    categories = [
        label_map[label]
        for label in valid_labels
    ]

    if len(categories) == 0:
        prompt = "This indoor scene contains no recognized object categories."
        return prompt, categories

    category_text = ", ".join(categories)

    prompt = (
        f"This indoor scene contains {category_text}."
    )

    return prompt, categories


def process_file(
    input_file,
    output_file,
    label_field,
    label_map
):
    """
    Generate a semantic-level prompt for one PLY file.
    """

    labels = load_semantic_labels(
        input_file,
        label_field=label_field
    )

    prompt, categories = generate_semantic_prompt(
        labels,
        label_map
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(prompt + "\n")

    return prompt, categories


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate semantic-level prompts from "
            "point cloud semantic segmentation results."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input PLY file or directory containing "
            "semantic segmentation results."
        )
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for generated semantic prompts."
    )

    parser.add_argument(
        "--label_field",
        default="semantic",
        help=(
            "Name of the semantic-label property in the PLY file. "
            "Default: semantic"
        )
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process a single file
    # --------------------------------------------------------
    if input_path.is_file():

        output_file = (
            output_dir /
            f"{input_path.stem}_semantic.txt"
        )

        prompt, categories = process_file(
            input_path,
            output_file,
            args.label_field,
            DEFAULT_LABEL_MAP
        )

        print(f"Input: {input_path}")
        print(f"Categories: {categories}")
        print(f"Prompt: {prompt}")
        print(f"Saved to: {output_file}")

    # --------------------------------------------------------
    # Process all PLY files in a directory
    # --------------------------------------------------------
    elif input_path.is_dir():

        files = sorted(
            input_path.glob("*.ply")
        )

        if len(files) == 0:
            print(
                f"No PLY files found in {input_path}"
            )
            return

        for file_id, input_file in enumerate(
            files,
            start=1
        ):

            print(
                f"\n[{file_id}/{len(files)}] "
                f"Processing {input_file.name}"
            )

            output_file = (
                output_dir /
                f"{input_file.stem}_semantic.txt"
            )

            prompt, categories = process_file(
                input_file,
                output_file,
                args.label_field,
                DEFAULT_LABEL_MAP
            )

            print(
                f"  Categories: {categories}"
            )

            print(
                f"  Prompt: {prompt}"
            )

    else:
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )


if __name__ == "__main__":
    main()
```
