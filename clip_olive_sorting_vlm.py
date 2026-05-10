#!/usr/bin/env python3

from clip_olives_vision_simulation import OliveVisionSimulation

import pybullet as p
import time


def main():

    print("🫒 CLIP Semantic Olive Sorting")
    print("=" * 50)

    print("Semantic classes:")
    print("• Unripe olives")
    print("• Ripe olives")
    print("• Partially ripe olives")
    print()

    sim = OliveVisionSimulation(
        gui_mode=True,
        model_name="openai/clip-vit-base-patch32"
    )

    try:

        print("Setting up simulation...")

        sim.setup_simulation()

        for _ in range(240):

            p.stepSimulation()

            time.sleep(1. / 240.)

        sim.set_initial_robot_pose()

        print("\nStarting CLIP semantic sorting...\n")

        results = sim.run_olive_sorting_demo()

        successful = sum(
            1 for r in results if r["success"]
        )

        accuracy = successful / len(results)

        print("\n" + "=" * 50)

        print("🫒 CLIP SORTING COMPLETE")

        print("=" * 50)

        print(f"Successful classifications: {successful}/{len(results)}")

        print(f"Accuracy: {accuracy:.3f}")

        print("\nPress ENTER to exit...")

        input()

    except KeyboardInterrupt:

        print("Interrupted by user.")

    finally:

        sim.cleanup()


if __name__ == "__main__":

    main()
