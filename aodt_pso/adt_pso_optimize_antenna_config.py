"""
ADT + PSO antenna configuration optimization (repo-native).

What this script is:
- A practical "aerial digital twin proxy" optimizer: it uses the repository's DT-style
  Scenario/RU/UE abstraction and a lightweight PHY model (FSPL + directional pattern)
  to let PSO optimize antenna configuration parameters efficiently.

What this script is NOT:
- A full NVIDIA Aerial Omniverse Digital Twin (AODT) / ray-tracing pipeline. Those
  require Omniverse + AODT backends; this script is designed to run locally while
  keeping the same optimization shape (decode -> simulate -> score).

Why this matches recent ADT practice (2024–2026):
- ADT papers and deployments commonly combine a DT (for environment-aware evaluation)
  with metaheuristics (PSO/GA) or differentiable simulators (e.g., Sionna) to optimize
  variables like placement, antenna orientation, and power under coverage/QoS goals.
  This script implements that "DT-in-the-loop optimization" pattern in a runnable form.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .aerial.dt import RU, UE, Scenario
from .aerial.phy import Antenna
from .optimizer.pso import PSO
from .simulation.signal_map import generate_ue_distribution


@dataclass(frozen=True)
class SearchSpace:
    area_bounds: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    ru_z_bounds: Tuple[float, float]
    freq_bounds_ghz: Tuple[float, float]
    elements_bounds: Tuple[int, int]
    azimuth_bounds_deg: Tuple[float, float]
    downtilt_bounds_deg: Tuple[float, float]

    def bounds_per_ru(self) -> List[Tuple[float, float]]:
        # Per-RU vector: x, y, z, freq, elements, azimuth, downtilt
        return [
            (self.area_bounds[0], self.area_bounds[1]),
            (self.area_bounds[2], self.area_bounds[3]),
            (self.ru_z_bounds[0], self.ru_z_bounds[1]),
            (self.freq_bounds_ghz[0], self.freq_bounds_ghz[1]),
            (float(self.elements_bounds[0]), float(self.elements_bounds[1])),
            (self.azimuth_bounds_deg[0], self.azimuth_bounds_deg[1]),
            (self.downtilt_bounds_deg[0], self.downtilt_bounds_deg[1]),
        ]


def decode_solution(solution: np.ndarray, num_rus: int, space: SearchSpace) -> List[RU]:
    params_per_ru = 7
    s = solution.reshape((num_rus, params_per_ru))
    rus: List[RU] = []
    for row in s:
        x, y, z, freq, elements_f, az, tilt = row.tolist()
        elements = int(round(elements_f))
        elements = int(np.clip(elements, space.elements_bounds[0], space.elements_bounds[1]))
        ant = Antenna(
            freq=float(freq),
            elements=elements,
            azimuth_deg=float(az),
            downtilt_deg=float(tilt),
        )
        rus.append(RU(position=np.array([x, y, z], dtype=float), antenna=ant))
    return rus


def objective_vectorized(
    particles: np.ndarray,
    ues: List[UE],
    num_rus: int,
    space: SearchSpace,
    coverage_floor_db: float,
    coverage_weight: float,
) -> np.ndarray:
    """
    Fitness to minimize.

    - Primary term: maximize sum(log10(best-signal)) across UEs (proxy QoS aggregate).
    - Optional penalty: encourage a minimum "coverage floor" (in dB) for more UEs.
    """
    num_particles = particles.shape[0]
    fitness = np.zeros(num_particles, dtype=float)

    cov_floor_linear = 10 ** (coverage_floor_db / 10.0)
    eps = 1e-18

    for i in range(num_particles):
        try:
            rus = decode_solution(particles[i], num_rus=num_rus, space=space)
            channel = Scenario(rus=rus, ues=ues).run()  # (U, R)
            best = np.max(channel, axis=1)  # (U,)

            # Aggregate "quality"
            quality = float(np.sum(np.log10(best + eps)))

            # Coverage encouragement: penalize fraction below floor
            below = np.mean(best < cov_floor_linear)
            penalty = coverage_weight * below

            fitness[i] = -quality + penalty
        except Exception:
            fitness[i] = 1e9

    return fitness


def build_bounds(num_rus: int, space: SearchSpace) -> List[Tuple[float, float]]:
    b = []
    per = space.bounds_per_ru()
    for _ in range(num_rus):
        b.extend(per)
    return b


def make_iter_logger(
    path: str,
    num_rus: int,
    space: SearchSpace,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "iteration",
                    "ru_id",
                    "x",
                    "y",
                    "z",
                    "freq_ghz",
                    "elements",
                    "azimuth_deg",
                    "downtilt_deg",
                    "gbest_fitness",
                ]
            )

    def _cb(iteration: int, gbest_pos: np.ndarray, gbest_fitness: float, _log: dict):
        rus = decode_solution(gbest_pos, num_rus=num_rus, space=space)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            for idx, ru in enumerate(rus):
                pos = ru.position
                ant = ru.antenna
                w.writerow(
                    [
                        iteration,
                        idx + 1,
                        f"{pos[0]:.3f}",
                        f"{pos[1]:.3f}",
                        f"{pos[2]:.3f}",
                        f"{ant.freq:.4f}",
                        int(ant.elements),
                        f"{ant.azimuth_deg:.2f}",
                        f"{ant.downtilt_deg:.2f}",
                        f"{gbest_fitness:.8f}",
                    ]
                )

    return _cb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-rus", type=int, default=3)
    p.add_argument("--num-ues", type=int, default=200)
    p.add_argument("--xmin", type=float, default=0.0)
    p.add_argument("--xmax", type=float, default=1000.0)
    p.add_argument("--ymin", type=float, default=0.0)
    p.add_argument("--ymax", type=float, default=1000.0)
    p.add_argument("--ru-zmin", type=float, default=10.0)
    p.add_argument("--ru-zmax", type=float, default=50.0)
    p.add_argument("--freq-min", type=float, default=2.0)
    p.add_argument("--freq-max", type=float, default=6.0)
    p.add_argument("--elem-min", type=int, default=1)
    p.add_argument("--elem-max", type=int, default=8)
    p.add_argument("--az-min", type=float, default=-180.0)
    p.add_argument("--az-max", type=float, default=180.0)
    p.add_argument("--tilt-min", type=float, default=0.0)
    p.add_argument("--tilt-max", type=float, default=20.0)

    p.add_argument("--ue-height", type=float, default=1.5)

    p.add_argument("--particles", type=int, default=60)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--topology", type=str, default="gbest", choices=["gbest", "lbest"])
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--coverage-floor-db", type=float, default=-120.0)
    p.add_argument("--coverage-weight", type=float, default=2.0)

    p.add_argument(
        "--log-csv",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adt_pso_antenna_log.csv"),
    )
    args = p.parse_args()

    np.random.seed(args.seed)

    space = SearchSpace(
        area_bounds=(args.xmin, args.xmax, args.ymin, args.ymax),
        ru_z_bounds=(args.ru_zmin, args.ru_zmax),
        freq_bounds_ghz=(args.freq_min, args.freq_max),
        elements_bounds=(args.elem_min, args.elem_max),
        azimuth_bounds_deg=(args.az_min, args.az_max),
        downtilt_bounds_deg=(args.tilt_min, args.tilt_max),
    )

    print("--- 1) Generating UE distribution ---")
    ues = generate_ue_distribution(
        num_ues=args.num_ues,
        area_bounds=space.area_bounds,
        ue_height=args.ue_height,
    )

    print("--- 2) Building PSO problem ---")
    bounds = build_bounds(args.num_rus, space)
    dims = len(bounds)
    obj = lambda swarm: objective_vectorized(
        swarm,
        ues=ues,
        num_rus=args.num_rus,
        space=space,
        coverage_floor_db=args.coverage_floor_db,
        coverage_weight=args.coverage_weight,
    )

    cb = make_iter_logger(args.log_csv, num_rus=args.num_rus, space=space)

    pso = PSO(
        objective_func=obj,
        num_dimensions=dims,
        bounds=bounds,
        num_particles=args.particles,
        max_iter=args.iters,
        topology=args.topology,
        # This logs generic (iteration, fitness, gbest JSON). We keep it off to avoid duplication.
        log_to_csv=None,
        on_iteration=cb,
    )

    print("--- 3) Running PSO ---")
    best_pos, best_fit, log = pso.run()

    print("\n--- 4) Best configuration ---")
    rus = decode_solution(best_pos, num_rus=args.num_rus, space=space)
    print(f"Best fitness: {best_fit:.6f}")
    for i, ru in enumerate(rus, start=1):
        x, y, z = ru.position.tolist()
        ant = ru.antenna
        print(
            f"RU-{i}: pos=({x:.1f},{y:.1f},{z:.1f}) "
            f"f={ant.freq:.2f}GHz elem={ant.elements} "
            f"az={ant.azimuth_deg:.1f}° tilt={ant.downtilt_deg:.1f}°"
        )

    print(f"\nWrote per-iteration best configs to: {args.log_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

