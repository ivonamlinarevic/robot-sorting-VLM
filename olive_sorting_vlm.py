# olive_sorting_vlm.py

#!/usr/bin/env python3
"""
Olive Sorting with Vision-Language Models (CLIP)

Conference-oriented robotic sorting demo using:
- PyBullet
- Panda robot
- CLIP zero-shot classification
- Procedurally generated olive-like objects
- Automatic color-based sorting

Classes:
- Green olives
- Black olives
- Mixed olives
"""

from olives_vision_simulation import OliveVisionSimulation
import pybullet as p
import time
import matplotlib.pyplot as plt


def main():
    print("🫒 Olive Sorting Demo")
    print("=" * 50)
    print("Classes:")
    print("• Green olives")
    print("• Black olives")
    print("• Mixed olives")
    print()

    sim = OliveVisionSimulation(
        gui_mode=True,
        model_name="openai/clip-vit-base-patch32",
        classification_method="hsv"
    )

    try:
        print("Setting up simulation...")
        sim.setup_simulation()

        for _ in range(240):
            p.stepSimulation()
            time.sleep(1. / 240.)

        sim.set_initial_robot_pose()

        print("Capturing initial scene...")
        image = sim.capture_camera_image()

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        ax.set_title("Initial Olive Sorting Scene")
        ax.axis('off')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        print("\nStarting automatic olive sorting...")

        results = sim.run_olive_sorting_demo()

        successful = sum(1 for r in results if r['success'])

        print("\n" + "=" * 50)
        print("🫒 SORTING COMPLETE")
        print("=" * 50)
        print(f"Successfully sorted: {successful}/{len(results)}")

        accuracy = successful / len(results)
        print(f"Accuracy: {accuracy:.3f}")

        print("\nPress ENTER to exit...")
        input()

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        sim.cleanup()


if __name__ == "__main__":
    main()
