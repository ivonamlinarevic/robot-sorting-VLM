#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import torch

from PIL import Image

from transformers import CLIPProcessor, CLIPModel


class OliveVisionSimulation:

    def __init__(self,
                 gui_mode=True,
                 model_name="openai/clip-vit-base-patch32"):

        # ============================================================
        # PYBULLET
        # ============================================================

        if gui_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ============================================================
        # CLIP
        # ============================================================

        self.device = "cpu"

        print(f"Loading CLIP model: {model_name}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            use_fast=False
        )

        print("CLIP loaded successfully")

        # ============================================================
        # ROBOT + OBJECTS
        # ============================================================

        self.robot_id = None
        self.table_id = None

        self.objects = {}

        self.grasp_constraint = None

        # ============================================================
        # CAMERA
        # ============================================================

        self.camera_position = [0.8, 0.0, 1.2]
        self.camera_target = [0.5, 0.0, 0.65]

        self.camera_width = 800
        self.camera_height = 480

        # ============================================================
        # ROBOT HOME
        # ============================================================

        self.home_position = [0.3, 0.0, 0.85]

        # ============================================================
        # SORTING ZONES
        # ============================================================

        table_z = 0.65
        zone_thickness = 0.02

        self.sorting_zones = {

            "green_zone": {
                "position": [0.0, 0.3, table_z - zone_thickness / 2],
                "color": [0.2, 0.8, 0.2, 0.5],
                "size": [0.3, 0.3, 0.02]
            },

            "black_zone": {
                "position": [0.0, -0.3, table_z - zone_thickness / 2],
                "color": [0.1, 0.1, 0.1, 0.5],
                "size": [0.3, 0.3, 0.02]
            },

            "mixed_zone": {
                "position": [0.8, 0.0, table_z - zone_thickness / 2],
                "color": [0.5, 0.4, 0.2, 0.5],
                "size": [0.3, 0.3, 0.02]
            }
        }

    # ============================================================
    # SIMULATION SETUP
    # ============================================================

    def setup_simulation(self):

        p.setGravity(0, 0, -9.81)

        p.loadURDF("plane.urdf")

        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.5, 0, 0.0],
            p.getQuaternionFromEuler([0, 0, 0]),
            globalScaling=1.0
        )

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [0.0, 0.0, 0.65],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )

        self.create_olive_objects()

        self.create_sorting_zones()

        self.setup_camera()

    # ============================================================
    # OLIVE GENERATION
    # ============================================================

    def create_olive_objects(self):

        print("Generating semantic olive dataset...")

        table_surface_z = 0.67

        olive_classes = [
            "unripe_olive",
            "ripe_olive",
            "partially_ripe_olive"
        ]

      #  positions = [
       #     [0.55, -0.15, table_surface_z],
        #    [0.60, -0.05, table_surface_z],
    #        [0.45, 0.10, table_surface_z],
     #       [0.50, 0.20, table_surface_z],
      #      [0.40, -0.20, table_surface_z],
       #     [0.65, 0.05, table_surface_z],
        #    [0.55, 0.00, table_surface_z],
         #   [0.45, -0.05, table_surface_z],
          #  [0.60, 0.15, table_surface_z]
        #]
        positions = []

        min_distance = 0.08

        while len(positions) < 9:

            x = random.uniform(0.38, 0.58)
            y = random.uniform(-0.18, 0.18)

            valid = True

            for pos in positions:

                dist = np.linalg.norm([
                    x - pos[0],
                    y - pos[1]
                ])

                if dist < min_distance:
                    valid = False
                    break

            if valid:

                positions.append([
                    x,
                    y,         
                    table_surface_z
                ])

        for i in range(9):

            olive_type = random.choice(olive_classes)

            size = random.uniform(0.06, 0.08)

            # ====================================================
            # UNRIPE OLIVE
            # ====================================================

            if olive_type == "unripe_olive":

                color = [
                    random.uniform(0.15, 0.25),
                    random.uniform(0.65, 0.90),
                    random.uniform(0.05, 0.15),
                    1
                ]

            # ====================================================
            # RIPE OLIVE
            # ====================================================

            elif olive_type == "ripe_olive":

                dark = random.uniform(0.02, 0.08)

                color = [
                    dark,
                    dark * 0.8,
                    dark * 1.2,
                    1
                ]

            # ====================================================
            # PARTIALLY RIPE OLIVE
            # ====================================================

            else:

                color = [
                    random.uniform(0.15, 0.28),
                    random.uniform(0.22, 0.38),
                    random.uniform(0.01, 0.04),
                    1
                ]

            config = {
                "name": f"{olive_type}_{i}",
                "olive_class": olive_type,
                "position": positions[i],
                "color": color,
                "size": size
            }

            object_id = self.create_olive(config)

            self.objects[config["name"]] = {
                "id": object_id,
                "config": config
            }

        print(f"Created {len(self.objects)} semantic olives")

    def create_olive(self, config):

        size = config["size"]

        collision_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=size / 3,
            height=size
        )

        visual_shape = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=size / 3,
            length=size,
            rgbaColor=config["color"],
            specularColor=[0.4, 0.4, 0.4]
        )

        object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=config["position"],
            baseOrientation=p.getQuaternionFromEuler([
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.3),
                random.uniform(0, np.pi)
            ])
        )
        
        # ADD DARK PATCH FOR PARTIALLY RIPE OLIVES

        if config["olive_class"] == "partially_ripe_olive":

            patch_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size / 5,
                rgbaColor=[0.05, 0.05, 0.05, 1]
            )

            patch_offset = [
                random.uniform(-0.015, 0.015),
                random.uniform(-0.015, 0.015),
                random.uniform(-0.015, 0.015)
            ]

            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=patch_visual,
                basePosition=[
                    config["position"][0] + patch_offset[0],
                    config["position"][1] + patch_offset[1],
                    config["position"][2] + patch_offset[2]
        ]
    )

        p.changeDynamics(
            object_id,
            -1,
            lateralFriction=2.0,
            rollingFriction=0.5,
            spinningFriction=0.5,
            restitution=0.0
        )

        return object_id

    # ============================================================
    # SORTING ZONES
    # ============================================================

    def create_sorting_zones(self):

        self.zone_ids = {}

        wall_height = 0.06
        wall_thickness = 0.01

        for zone_name, zone_config in self.sorting_zones.items():

            pos = zone_config["position"]
            color = zone_config["color"]

            size_x = zone_config["size"][0]
            size_y = zone_config["size"][1]
            size_z = zone_config["size"][2]

            # ====================================================
            # FLOOR
            # ====================================================

            floor_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    size_x / 2,
                    size_y / 2,
                    size_z / 2
                ]
            )

            floor_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    size_x / 2,
                    size_y / 2,
                    size_z / 2
                ],
                rgbaColor=color
            )

            floor_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=floor_collision,
                baseVisualShapeIndex=floor_visual,
                basePosition=pos
            )    

            self.zone_ids[f"{zone_name}_floor"] = floor_id

            # ====================================================
            # WALLS    
            # ====================================================

            wall_collision_x = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    wall_thickness / 2,
                    size_y / 2,
                    wall_height / 2
                ]
            )    

            wall_visual_x = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    wall_thickness / 2,
                    size_y / 2,
                    wall_height / 2
                ],
                rgbaColor=color
            )

            wall_collision_y = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    size_x / 2,
                    wall_thickness / 2,
                    wall_height / 2
                ]
            )

            wall_visual_y = p.createVisualShape(
                p.GEOM_BOX,    
                halfExtents=[
                    size_x / 2,
                    wall_thickness / 2,
                    wall_height / 2
                ],
                rgbaColor=color
            )

            wall_z = pos[2] + wall_height / 2

            # LEFT WALL
            p.createMultiBody(
                baseMass=0,    
                baseCollisionShapeIndex=wall_collision_x,
                baseVisualShapeIndex=wall_visual_x,
                basePosition=[    
                    pos[0] - size_x / 2,
                    pos[1],
                    wall_z
                ]
            )    

            # RIGHT WALL
            p.createMultiBody(
                baseMass=0,    
                baseCollisionShapeIndex=wall_collision_x,
                baseVisualShapeIndex=wall_visual_x,
                basePosition=[
                    pos[0] + size_x / 2,
                    pos[1],
                    wall_z
                ]
            )

            # FRONT WALL
            p.createMultiBody(
                baseMass=0,    
                baseCollisionShapeIndex=wall_collision_y,
                baseVisualShapeIndex=wall_visual_y,    
                basePosition=[
                    pos[0],
                    pos[1] - size_y / 2,
                    wall_z
                ]
            )

            # BACK WALL
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_collision_y,
                baseVisualShapeIndex=wall_visual_y,
                basePosition=[
                    pos[0],
                    pos[1] + size_y / 2,    
                    wall_z
                ]
            )

    # ============================================================
    # CAMERA
    # ============================================================

    def setup_camera(self):

        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0.5, 0, 0.5]
        )

    def capture_camera_image_with_segmentation(self):

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 0, 1]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100.0
        )

        width, height, rgb_array, depth_array, seg_array = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        rgb_array = rgb_array[:, :, :3]

        image = Image.fromarray(rgb_array, 'RGB')

        return image, depth_array, seg_array

    # ============================================================
    # SEGMENTATION CROPS
    # ============================================================

    def get_masked_object_crop(self, object_name):

        image, depth, seg = self.capture_camera_image_with_segmentation()

        rgb = np.array(image)

        object_id = self.objects[object_name]["id"]

        mask = seg == object_id

        if np.sum(mask) == 0:
            return image

        ys, xs = np.where(mask)

        x1 = np.min(xs)
        x2 = np.max(xs)

        y1 = np.min(ys)
        y2 = np.max(ys)

        padding = 10

        x1 = max(0, x1 - padding)
        x2 = min(self.camera_width, x2 + padding)

        y1 = max(0, y1 - padding)
        y2 = min(self.camera_height, y2 + padding)

        cropped = rgb[y1:y2, x1:x2]

        cropped_mask = mask[y1:y2, x1:x2]

        result = np.zeros_like(cropped)

        result[cropped_mask] = cropped[cropped_mask]

        return Image.fromarray(result)

    # ============================================================
    # CLIP CLASSIFICATION
    # ============================================================

    def classify_olive_clip(self, crop):

        prompts = {

            "ripe_olive":
                "a photo of a ripe black olive",

            "unripe_olive":
                "a photo of an unripe green olive",

            "partially_ripe_olive":
                "a photo of a green and black olive with mixed colors"
        }

        scores = {}

        for label, prompt in prompts.items():

            inputs = self.processor(
                text=[prompt],
                images=[crop],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():

                outputs = self.model(**inputs)

                similarity = outputs.logits_per_image[0][0].item()

            scores[label] = similarity

        predicted = max(scores.items(), key=lambda x: x[1])[0]

        return predicted, scores

    # ============================================================
    # MAIN CLASSIFIER
    # ============================================================

    def classify_olive(self, object_name):

        crop = self.get_masked_object_crop(object_name)

        predicted, scores = self.classify_olive_clip(crop)

        return predicted, scores

    # ============================================================
    # ROBOT CONTROL
    # ============================================================

    def set_initial_robot_pose(self):

        initial_joint_positions = [
            0,
            -0.785,
            0,
            -2.356,
            0,
            1.571,
            0.785
        ]

        for i in range(7):

            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=initial_joint_positions[i],
                force=500
            )

        for _ in range(480):

            p.stepSimulation()

            time.sleep(1. / 240.)

    def move_to_position(self, target_position):

        target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorLinkIndex=11,
            targetPosition=target_position,
            targetOrientation=target_orientation
        )

        for i in range(7):

            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500
            )

        self.wait_for_movement()

    def wait_for_movement(self, timeout=3.0):

        start_time = time.time()

        while time.time() - start_time < timeout:

            p.stepSimulation()

            time.sleep(1. / 240.)

    def control_gripper(self, position):

        for joint in [9, 10]:

            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=position,
                force=200
            )

        for _ in range(120):

            p.stepSimulation()

            time.sleep(1. / 240.)

    # ============================================================
    # PICK OBJECT
    # ============================================================

    def pick_object(self, object_name):

        obj_info = self.objects[object_name]

        object_id = obj_info["id"]

        obj_pos, _ = p.getBasePositionAndOrientation(object_id)

        print(f"Picking up: {object_name}")

        self.control_gripper(0.04)

        approach = [
            obj_pos[0],
            obj_pos[1],
            obj_pos[2] + 0.20
        ]

        self.move_to_position(approach)

        grasp = [
            obj_pos[0],
            obj_pos[1],
            obj_pos[2] + 0.02
        ]

        self.move_to_position(grasp)

        self.control_gripper(0.0)

        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=11,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.12],
            childFramePosition=[0, 0, 0]
        )

        print("Constraint grasp attached")

        lift = [
            obj_pos[0],
            obj_pos[1],
            obj_pos[2] + 0.35
        ]

        self.move_to_position(lift)

        return True

    # ============================================================
    # PLACE OBJECT
    # ============================================================

    def place_object_in_zone(self,
                             object_name,
                             predicted_class):

        zone_mapping = {

            "unripe_olive": "green_zone",

            "ripe_olive": "black_zone",

            "partially_ripe_olive": "mixed_zone"
        }

        zone_name = zone_mapping[predicted_class]

        zone_position = self.sorting_zones[zone_name]["position"]

        target = [
            zone_position[0],
            zone_position[1],
            zone_position[2] + 0.15
        ]

        self.move_to_position(target)

        if self.grasp_constraint is not None:

            p.removeConstraint(self.grasp_constraint)

            self.grasp_constraint = None

        self.control_gripper(0.04)

        retreat = [
            zone_position[0],
            zone_position[1],
            zone_position[2] + 0.25
        ]

        self.move_to_position(retreat)

        self.move_to_position(self.home_position)

    # ============================================================
    # MAIN DEMO
    # ============================================================

    def run_olive_sorting_demo(self):

        results = []

        for object_name in self.objects.keys():

            print("\n" + "-" * 50)

            print(f"Processing: {object_name}")

            predicted_class, scores = self.classify_olive(object_name)

            true_class = self.objects[object_name]["config"]["olive_class"]

            print(f"True class: {true_class}")

            print(f"Predicted class: {predicted_class}")

            print(f"CLIP scores: {scores}")

            success = predicted_class == true_class

            self.pick_object(object_name)

            self.place_object_in_zone(
                object_name,
                predicted_class
            )

            results.append({

                "object": object_name,

                "true_class": true_class,

                "predicted_class": predicted_class,

                "success": success,

                "scores": scores
            })

            print(f"Success: {success}")

        return results

    # ============================================================
    # CLEANUP
    # ============================================================

    def cleanup(self):

        p.disconnect()

        print("Simulation ended")
