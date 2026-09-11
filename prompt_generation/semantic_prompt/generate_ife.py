```python
import argparse
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from implicit_feature_encoding import ImplicitFeatureEncoding


def load_ply(filename):
    """Load XYZ coordinates from a PLY file."""
    ply = PlyData.read(filename)
    vertex = ply["vertex"].data

    points = np.column_stack([
        vertex["x"],
        vertex["y"],
        vertex["z"]
    ]).astype(np.float32)

    return points


def save_augmented_ply(filename, cloud, implicit_features):
    """
    Save the input point cloud together with implicit features.

    Output attributes:
        x, y, z
        d1, d2, ..., dM
    """

    num_points = len(cloud)
    feature_dim = implicit_features.shape[1]

    vertex_data = np.empty(
        num_points,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4")
        ] + [
            (f"d{i + 1}", "f4")
            for i in range(feature_dim)
        ]
    )

    vertex_data["x"] = cloud[:, 0]
    vertex_data["y"] = cloud[:, 1]
    vertex_data["z"] = cloud[:, 2]

    for i in range(feature_dim):
        vertex_data[f"d{i + 1}"] = implicit_features[:, i]

    vertex_element = PlyElement.describe(
        vertex_data,
        "vertex"
    )

    PlyData(
        [vertex_element],
        text=False
    ).write(filename)


def main():

    parser = argparse.ArgumentParser(
        description="Generate implicit features for point clouds."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing input PLY files."
    )

    parser.add_argument(
        "--sample_sphere",
        required=True,
        help="PLY file containing the predefined sample sphere."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for augmented point clouds."
    )

    parser.add_argument(
        "--neighbor_radius",
        type=float,
        default=0.15,
        help="Radius for local neighborhood searching."
    )

    parser.add_argument(
        "--fix_z_axis",
        type=int,
        default=1,
        help="Use fixed-Z-axis canonicalization (1) or full 3D canonicalization (0)."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Load sample sphere
    # ------------------------------------------------------------
    sample_sphere = load_ply(
        args.sample_sphere
    )

    ife = ImplicitFeatureEncoding(
        neighbor_radius=args.neighbor_radius,
        fix_z_axis=(args.fix_z_axis != 0)
    )

    ife.set_sample_sphere(
        sample_sphere
    )

    print("CALCULATION SETTINGS")
    print(
        f"  (1) Neighbor radius: "
        f"{ife.get_neighbor_radius():.4f} m"
    )
    print(
        f"  (2) Sample sphere radius: "
        f"{ife.get_sample_radius():.4f} m"
    )
    print(
        f"  (3) Implicit feature dimension: "
        f"{ife.get_sample_size()}"
    )

    files = sorted(
        input_dir.glob("*.ply")
    )

    total_points = 0
    total_calculation_time = 0.0
    total_time = 0.0

    print("\nSTART BATCH PROCESSING")

    for file_id, input_file in enumerate(files, 1):

        print(
            f"\n[{file_id}/{len(files)}] "
            f"Processing {input_file.name} ..."
        )

        output_file = (
            output_dir /
            input_file.name
        )

        if output_file.exists():
            print(
                "  Output file already exists. Skipping..."
            )
            continue

        # --------------------------------------------------------
        # Load point cloud
        # --------------------------------------------------------
        tic = time.perf_counter()

        cloud = load_ply(
            input_file
        )

        toc = time.perf_counter()

        print(
            f"  (1) Load {len(cloud)} points "
            f"in {toc - tic:.3f} seconds."
        )

        total_points += len(cloud)

        # --------------------------------------------------------
        # Calculate implicit features
        # --------------------------------------------------------
        tic = time.perf_counter()

        ife.set_input_cloud(
            cloud
        )

        implicit_features = (
            ife.calculate_implicit_features()
        )

        toc = time.perf_counter()

        calculation_time = toc - tic

        print(
            f"  (2) Calculate implicit features "
            f"in {calculation_time:.3f} seconds."
        )

        total_calculation_time += (
            calculation_time
        )

        # --------------------------------------------------------
        # Save augmented point cloud
        # --------------------------------------------------------
        tic = time.perf_counter()

        save_augmented_ply(
            output_file,
            cloud,
            implicit_features
        )

        toc = time.perf_counter()

        save_time = toc - tic

        print(
            f"  (3) Save augmented point cloud "
            f"in {save_time:.3f} seconds."
        )

        total_time += (
            calculation_time + save_time
        )

    print("\nEND BATCH PROCESSING")

    print(
        f"  (1) Total number of points: "
        f"{total_points}"
    )

    if total_calculation_time > 60:
        print(
            f"  (2) Implicit calculation time: "
            f"{total_calculation_time / 60:.2f} minutes"
        )
    else:
        print(
            f"  (2) Implicit calculation time: "
            f"{total_calculation_time:.2f} seconds"
        )

    if total_time > 60:
        print(
            f"  (3) Total processing time: "
            f"{total_time / 60:.2f} minutes"
        )
    else:
        print(
            f"  (3) Total processing time: "
            f"{total_time:.2f} seconds"
        )


if __name__ == "__main__":
    main()
```
