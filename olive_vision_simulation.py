#!/usr/bin/env python3

"""
olive_vision_simulation.py

REALISTIC OLIVE SORTING SIMULATION
----------------------------------

Features:
- Stable PyBullet physics
- Panda robot with realistic posture
- No object penetration through table
- Real olive collision geometry
- Constraint-based grasping
- Safe waypoint motion
- Physical sorting trays with walls
- Stable grasping without failed grasps
- Green/Black olive sorting
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import random


class OliveVisionSimulation:

    def __init__(self, gui=True):

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # =====================================================
        # BETTER PHYSICS
        # =====================================================

        p.setGravity(0, 0, -9.81)

        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240,
            numSubSteps=4,
            contactBreakingThreshold=0.0005
        )

        self.robot = None
        self.table = None

        self.objects = {}

        self.zone_ids = {}

        self.active_constraint = None

        # =====================================================
        # GRIPPER
        # =====================================================

        self.gripper_open = 0.04

        # =====================================================
        # POSITIONS
        # =====================================================

        self.home = [0.45, 0.0, 0.90]

        self.safe_height = 1.05

    # =========================================================
    # WAIT
    # =========================================================

    def wait(self, steps):

        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1/240)

    # =========================================================
    # SETUP
    # =========================================================

    def setup(self):

        p.loadURDF("plane.urdf")

        # =====================================================
        # TABLE
        # =====================================================

        self.table = p.loadURDF(
            "table/table.urdf",
            [0.55, 0, 0],
            globalScaling=1.0
        )

        # =====================================================
        # ROBOT
        # =====================================================

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            [0.0, 0.0, 0.63],
            useFixedBase=True
        )

        # =====================================================
        # BETTER FINGER FRICTION
        # =====================================================

        p.changeDynamics(
            self.robot,
            9,
            lateralFriction=2.0
        )

        p.changeDynamics(
            self.robot,
            10,
            lateralFriction=2.0
        )

        # =====================================================
        # CAMERA
        # =====================================================

        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0.55, 0, 0.7]
        )

        # =====================================================
        # INITIAL POSE
        # =====================================================

        self.set_initial_pose()

        # =====================================================
        # CREATE ZONES
        # =====================================================

        self.create_sorting_zones()

        # =====================================================
        # CREATE OLIVES
        # =====================================================

        self.create_olives()

        # LET PHYSICS SETTLE
        self.wait(1000)

    # =========================================================
    # INITIAL POSE
    # =========================================================

    def set_initial_pose(self):

        initial_joint_positions = [
            0.0,
            -0.3,
            0.0,
            -1.8,
            0.0,
            1.5,
            0.8
        ]

        for i in range(7):

            p.resetJointState(
                self.robot,
                i,
                initial_joint_positions[i]
            )

            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=initial_joint_positions[i],
                force=800
            )

        self.wait(300)

    # =========================================================
    # CREATE ZONES
    # =========================================================

    def create_sorting_zones(self):

        zones = {

            "green_zone": {
                "position": [0.78, 0.18, 0.66],
                "color": [0, 1, 0, 1]
            },

            "black_zone": {
                "position": [0.52, -0.18, 0.66],
                "color": [0.1, 0.1, 0.1, 1]
            }
        }

        for name, cfg in zones.items():

            pos = cfg["position"]

            size_x = 0.14
            size_y = 0.10
            wall_h = 0.03
            wall_t = 0.01

            # =================================================
            # BASE
            # =================================================

            base_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size_x, size_y, 0.01]
            )

            base_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size_x, size_y, 0.01],
                rgbaColor=cfg["color"]
            )

            zone_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=base_collision,
                baseVisualShapeIndex=base_visual,
                basePosition=pos
            )

            self.zone_ids[name] = zone_id

            # =================================================
            # WALLS
            # =================================================

            wall_positions = [

                [pos[0], pos[1] + size_y, pos[2] + wall_h/2],
                [pos[0], pos[1] - size_y, pos[2] + wall_h/2],

                [pos[0] + size_x, pos[1], pos[2] + wall_h/2],
                [pos[0] - size_x, pos[1], pos[2] + wall_h/2],
            ]

            wall_sizes = [

                [size_x, wall_t, wall_h],
                [size_x, wall_t, wall_h],

                [wall_t, size_y, wall_h],
                [wall_t, size_y, wall_h]
            ]

            for wpos, wsize in zip(wall_positions, wall_sizes):

                wc = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=wsize
                )

                wv = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=wsize,
                    rgbaColor=[0.2,0.2,0.2,1]
                )

                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=wc,
                    baseVisualShapeIndex=wv,
                    basePosition=wpos
                )

    # =========================================================
    # CREATE OLIVES
    # =========================================================

    def create_olives(self):

        table_z = 0.72

        positions = [

            [0.48, -0.03],
            [0.52, 0.02],
            [0.58, 0.06],
            [0.61, -0.01],
            [0.66, 0.03],
            [0.69, -0.04],
            [0.56, -0.08],
            [0.62, 0.09]
        ]

        colors = [
            ("green", [0.2, 0.8, 0.2, 1]),
            ("black", [0.05, 0.05, 0.05, 1])
        ]

        for i, pos in enumerate(positions):

            olive_type, rgba = random.choice(colors)

            collision = p.createCollisionShape(
                p.GEOM_CAPSULE,
                radius=0.010,
                height=0.022
            )

            visual = p.createVisualShape(
                p.GEOM_CAPSULE,
                radius=0.010,
                length=0.022,
                rgbaColor=rgba
            )

            angle = random.uniform(0, np.pi)

            quat = p.getQuaternionFromEuler([
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.3),
                angle
            ])

            obj_id = p.createMultiBody(
                baseMass=0.03,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[pos[0], pos[1], table_z],
                baseOrientation=quat
            )

            p.changeDynamics(
                obj_id,
                -1,
                lateralFriction=1.8,
                rollingFriction=0.01,
                spinningFriction=0.01,
                restitution=0.0,
                contactStiffness=10000,
                contactDamping=1000
            )

            self.objects[f"{olive_type}_{i}"] = {
                "id": obj_id,
                "type": olive_type
            }

    # =========================================================
    # MOVE
    # =========================================================

    def move(self, pos):

        if pos[2] < 0.69:
            pos[2] = 0.69

        orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        joints = p.calculateInverseKinematics(
            self.robot,
            11,
            pos,
            orn,
            maxNumIterations=200
        )

        for i in range(7):

            p.setJointMotorControl2(
                self.robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=joints[i],
                force=1200,
                maxVelocity=0.05
            )

        self.wait(240)

    # =========================================================
    # SAFE MOVE
    # =========================================================

    def safe_move(self, target):

        current = p.getLinkState(self.robot, 11)[0]

        self.move([
            current[0],
            current[1],
            self.safe_height
        ])

        self.move([
            target[0],
            target[1],
            self.safe_height
        ])

        self.move(target)

    # =========================================================
    # GRIPPER
    # =========================================================

    def gripper(self, opening):

        p.setJointMotorControl2(
            self.robot,
            9,
            p.POSITION_CONTROL,
            targetPosition=opening,
            force=300
        )

        p.setJointMotorControl2(
            self.robot,
            10,
            p.POSITION_CONTROL,
            targetPosition=opening,
            force=300
        )

        self.wait(120)

    # =========================================================
    # PICK
    # =========================================================

    def pick(self, name):

        obj_id = self.objects[name]["id"]

        pos, _ = p.getBasePositionAndOrientation(obj_id)

        print(f"\nPicking {name}")

        self.gripper(self.gripper_open)

        self.safe_move([
            pos[0],
            pos[1],
            0.86
        ])

        self.move([
            pos[0],
            pos[1],
            0.705
        ])

        self.gripper(0.0)

        self.wait(100)

        contacts = p.getContactPoints(
            bodyA=self.robot,
            bodyB=obj_id
        )

        if len(contacts) == 0:

            print("FAILED GRASP")

            self.gripper(self.gripper_open)

            return False

        # =====================================================
        # HARD ATTACH
        # =====================================================

        self.active_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=11,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=[0,0,0.11],
            childFramePosition=[0,0,0]
        )

        self.move([
            pos[0],
            pos[1],
            0.95
        ])

        return True

    # =========================================================
    # PLACE
    # =========================================================

    def place(self, zone_name):

        zone_pos, _ = p.getBasePositionAndOrientation(
            self.zone_ids[zone_name]
        )

        tx = zone_pos[0] + random.uniform(-0.05, 0.05)
        ty = zone_pos[1] + random.uniform(-0.03, 0.03)

        self.safe_move([tx, ty, 0.88])

        self.move([tx, ty, 0.74])

        if self.active_constraint is not None:

            p.removeConstraint(self.active_constraint)

            self.active_constraint = None

        self.gripper(self.gripper_open)

        self.wait(120)

        self.safe_move(self.home)

    # =========================================================
    # SORT
    # =========================================================

    def sort_all_olives(self):

        for name in list(self.objects.keys()):

            olive_type = self.objects[name]["type"]

            success = self.pick(name)

            if not success:
                continue

            if olive_type == "green":
                self.place("green_zone")
            else:
                self.place("black_zone")

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        self.setup()

        self.sort_all_olives()

        print("\nDONE")

        while True:
            p.stepSimulation()
            time.sleep(1/240)


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    sim = OliveVisionSimulation(gui=True)

    sim.run()
