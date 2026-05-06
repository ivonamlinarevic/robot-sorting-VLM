#!/usr/bin/env python3

"""
olives_vlm.py

Launcher file for the olive sorting simulation.
"""

from olive_vision_simulation import OliveVisionSimulation
import pybullet as p
import time


def main():

    print("\n🫒 OLIVE SORTING SIMULATION")
    print("=" * 50)

    sim = None

    try:

        # =====================================================
        # CREATE SIMULATION
        # =====================================================

        sim = OliveVisionSimulation(gui=True)

        print("\nCreating simulation world...")

        sim.setup()

        # =====================================================
        # LET PHYSICS STABILIZE
        # =====================================================

        print("Stabilizing physics...")

        for _ in range(480):

            p.stepSimulation()

            time.sleep(1.0 / 240.0)

        # =====================================================
        # START SORTING
        # =====================================================

        print("\nStarting olive sorting...")

        sim.sort_all_olives()

        print("\n✅ SORTING COMPLETE")

        print("\nSimulation still running.")
        print("Close the PyBullet window or press CTRL+C to exit.")

        # =====================================================
        # KEEP GUI OPEN
        # =====================================================

        while True:

            p.stepSimulation()

            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:

        print("\nSimulation stopped by user.")

    except Exception as e:

        print("\n❌ ERROR:", e)

    finally:

        if sim is not None:

            sim.cleanup()


if __name__ == "__main__":

    main()
