import os
import subprocess
import sys
from typing import List, Tuple


def run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def drift_to_flops_budget(drift: float, base_flops: float = 41e8) -> float:
    # Example policy: as drift increases, allow slightly larger FLOPs to adapt
    scale = 1.0 + 0.2 * drift  # up to +20%
    return base_flops * scale


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(project_root, "configs", "classification", "R50_FLOPs.py")

    # Simulated concept evolution over three time steps
    steps: List[Tuple[str, float]] = [
        ("step0", 0.0),
        ("step1", 0.3),
        ("step2", 0.6),
    ]

    summary_rows: List[Tuple[str, float, float, str, str]] = []

    for step_name, drift in steps:
        flops_budget = drift_to_flops_budget(drift)
        work_dir = os.path.join(project_root, "save_model", f"concept_{step_name}")

        # Tiny/fast search settings for demo purposes
        search_overrides = [
            f"work_dir={work_dir}",
            f"budgets.0.budget={flops_budget}",
            "search.num_random_nets=10",
            "search.popu_size=4",
        ]

        run([
            sys.executable,
            os.path.join(project_root, "tools", "search.py"),
            cfg_path,
            "--cfg_options",
            *search_overrides,
        ])

        export_out = os.path.join(project_root, "output_dir")
        run([
            sys.executable,
            os.path.join(project_root, "tools", "export.py"),
            work_dir,
            export_out,
        ])

        exported_dir = os.path.join(export_out, os.path.basename(work_dir))
        best_txt = os.path.join(work_dir, "best_structure.txt")
        best_json = os.path.join(exported_dir, "best_structure.json")
        summary_rows.append((step_name, drift, flops_budget, best_txt, best_json))

    print("\nConcept evolution NAS summary:")
    print("step\tdrift\tFLOPs_budget\tbest_structure.txt\tbest_structure.json")
    for row in summary_rows:
        step_name, drift, flops_budget, best_txt, best_json = row
        print(f"{step_name}\t{drift:.2f}\t{flops_budget:.1f}\t{best_txt}\t{best_json}")


if __name__ == "__main__":
    main()


