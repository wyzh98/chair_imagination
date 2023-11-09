import pybullet as p
import pybullet_data
import numpy as np
import math
import os
import trimesh
import time
from imagination import ImaginationMatrix


class ChairFunctionalityPrecheck:
    def __init__(self, cube_urdf, check_process=False, mp4_dir=None):
        self.cube_urdf = cube_urdf
        self.check_process = check_process
        self.mp4_dir = mp4_dir

        self.num_x_throw_cube = 3
        self.num_y_throw_cube = 3
        self.chair_adj_dist = 4  # distance between two adjacent chairs
        self.throw_directions = 8
        self.simulation_iter = 500

        self.chair_id_list = []
        self.cube_id_list = []

        self.chair_friction_coeff = 1.0

    def get_posterior_distrib(self, obj_urdf, obj_transform_mesh, stable_orn, stable_pos):
        if self.check_process:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(3, 90, -45, [3, 2, 0])

        # Load plane
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, restitution=0.0)

        # Load chair
        trimesh_mesh = trimesh.load(obj_transform_mesh)
        chair_extents = trimesh_mesh.extents
        chair_extents_argsort = np.argsort(np.array(chair_extents))
        chair_scale = chair_extents[chair_extents_argsort[1]]
        cube_scale = chair_scale / 8

        # Place chairs
        for x_idx in range(self.num_x_throw_cube):
            for y_idx in range(self.num_y_throw_cube):
                chair_start_x = x_idx * self.chair_adj_dist
                chair_start_y = y_idx * self.chair_adj_dist

                # Set the OBB center to positions
                chair_id = p.loadURDF(
                    obj_urdf,
                    basePosition=[chair_start_x, chair_start_y, 0.0],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
                p.changeDynamics(chair_id, -1, restitution=0.0, mass=0)  # set chair static
                p.changeDynamics(chair_id, -1, lateralFriction=self.chair_friction_coeff)
                self.chair_id_list.append(chair_id)

        # Place cubes
        for x_idx in range(self.num_x_throw_cube):
            for y_idx in range(self.num_y_throw_cube):
                chair_start_x = x_idx * self.chair_adj_dist
                chair_start_y = y_idx * self.chair_adj_dist

                cube_id = p.loadURDF(self.cube_urdf,
                                     basePosition=[chair_start_x, chair_start_y, 0.2],
                                     globalScaling=cube_scale)
                p.changeDynamics(cube_id, -1, restitution=0.0, lateralFriction=10.0)
                self.cube_id_list.append(cube_id)

        # Get chair initial location
        chair_id = p.loadURDF(obj_urdf)
        chair_curr_pos, _ = p.getBasePositionAndOrientation(chair_id)
        chair_curr_pos = np.transpose(np.array(chair_curr_pos))
        p.removeBody(chair_id)

        # Save video
        if (self.mp4_dir is not None) and self.check_process:
            obj_name = obj_urdf.split('/')[-1].split('.')[0]
            mp4_file_name = obj_name + "_chair_precheck.mp4"
            mp4_file_path = os.path.join(self.mp4_dir, mp4_file_name)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, mp4_file_path)

        # Throw cube to the chair
        stable_info = dict()
        for chair_stable_idx, chair_stable_orn in enumerate(stable_orn):
            stable_info[chair_stable_idx] = []
            chair_stable_pos = stable_pos[chair_stable_idx]
            chair_start_pos = [0.0, 0.0, chair_stable_pos[-1]]
            p.setGravity(0, 0, -10)

            for x_idx in range(self.num_x_throw_cube):
                for y_idx in range(self.num_y_throw_cube):
                    throw_idx = y_idx + x_idx * self.num_y_throw_cube
                    if throw_idx != self.num_x_throw_cube * self.num_y_throw_cube - 1:  # Throw from side
                        chair_z_axis_angle = throw_idx * 2 * math.pi / self.throw_directions
                    else:  # Throw from top
                        chair_z_axis_angle = 0

                    chair_start_orn = [chair_stable_orn[0], chair_stable_orn[1], chair_z_axis_angle]
                    chair_stable_orn_mat = np.reshape(np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(chair_start_orn))), (3, 3))
                    chair_initial_center = np.copy(chair_curr_pos)
                    chair_center = np.dot(chair_stable_orn_mat, chair_initial_center)
                    chair_start_pos[0] = chair_center[0] + x_idx * self.chair_adj_dist
                    chair_start_pos[1] = chair_center[1] + y_idx * self.chair_adj_dist

                    p.resetBasePositionAndOrientation(self.chair_id_list[throw_idx],
                                                      chair_start_pos,
                                                      p.getQuaternionFromEuler(chair_start_orn))

                    chair_aabb = p.getAABB(self.chair_id_list[throw_idx])
                    chair_bbox_largest = chair_aabb[1][2]

                    # Cube
                    cube_id = self.cube_id_list[throw_idx]
                    if throw_idx != self.num_x_throw_cube * self.num_y_throw_cube - 1:
                        cube_dx = (chair_bbox_largest + 0.15) * math.sqrt(2) / 2
                        cube_z = (chair_bbox_largest + 0.15) * math.sqrt(2) / 2
                        cube_start_vel_x = -math.sqrt(1 * 10 * cube_z)
                    else:
                        cube_dx = 0
                        cube_z = chair_bbox_largest + 0.15
                        cube_start_vel_x = 0
                    cube_start_pos = (x_idx * self.chair_adj_dist + cube_dx, y_idx * self.chair_adj_dist, cube_z)
                    cube_start_orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
                    p.resetBasePositionAndOrientation(cube_id, cube_start_pos, cube_start_orn)
                    p.resetBaseVelocity(cube_id, [cube_start_vel_x, 0, 0])

            for i in range(int(self.simulation_iter)):
                p.stepSimulation()

            # Check each sitting
            for x_idx in range(self.num_x_throw_cube):
                for y_idx in range(self.num_y_throw_cube):
                    throw_id = y_idx + x_idx * self.num_y_throw_cube
                    cube_id = self.cube_id_list[throw_id]
                    chair_id = self.chair_id_list[throw_id]
                    cube_affordance = self.check_cube_on_chair(cube_id, chair_id)
                    if cube_affordance:
                        stable_info[chair_stable_idx].append(throw_id)
        p.disconnect()
        print(f"Cube Stable Info: {stable_info}")
        return stable_info

    def check_cube_on_chair(self, cube_id, chair_id):
        cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
        n_contact_pt = len(p.getContactPoints(cube_id, chair_id, -1))
        if cube_pos[2] > 0.2 and n_contact_pt > 0:
            return True
        else:
            return False
