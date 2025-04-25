# environment.py
import gym
import numpy as np
import pybullet as p
import pybullet_data

class CubeTableEnv(gym.Env):
    def __init__(self):
        super(CubeTableEnv, self).__init__()
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, 0])
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.5])
        self.objects = self.setup_objects()
        self.camera = self.setup_camera()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,))  # 7 joints + gripper
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(240, 320, 3))
        self.initial_positions = {}  # Track initial object positions

    def setup_objects(self):
        objects = [
            {"id": p.loadURDF("cube.urdf", [0.5, 0, 0.7], globalScaling=0.05), "name": "red cube", "color": "red", "type": "cube"},
            {"id": p.loadURDF("cube.urdf", [0.5, 0.1, 0.7], globalScaling=0.05), "name": "blue cube", "color": "blue", "type": "cube"},
            {"id": p.loadURDF("cube.urdf", [0.5, -0.1, 0.7], globalScaling=0.05), "name": "green cube", "color": "green", "type": "cube"},
            {"id": p.loadURDF("tetrahedron.urdf", [0.6, 0, 0.7], globalScaling=0.05), "name": "orange pyramid", "color": "orange", "type": "pyramid"},
            {"id": p.loadURDF("tetrahedron.urdf", [0.6, 0.1, 0.7], globalScaling=0.05), "name": "yellow pyramid", "color": "yellow", "type": "pyramid"},
            {"id": p.loadURDF("cube.urdf", [0.7, 0, 0.7], globalScaling=0.1), "name": "box", "color": None, "type": "box"},
        ]
        return objects

    def setup_camera(self):
        return p.computeViewMatrix([0, 0, 1], [0.6, 0, 0.7], [0, 0, 1])

    def get_observation(self):
        img = p.getCameraImage(320, 240, viewMatrix=self.camera)[2]
        robot_state = [state[0] for state in p.getJointStates(self.robot_id, range(7))]
        return np.array(img), np.array(robot_state)

    def step(self, action):
        # Store initial positions before action
        self.initial_positions = {obj["name"]: p.getBasePositionAndOrientation(obj["id"])[0] for obj in self.objects}
        
        # Execute action
        p.setJointMotorControlArray(self.robot_id, range(7), p.POSITION_CONTROL, targetPositions=action[:7])
        p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=action[7])
        p.stepSimulation()
        obs, state = self.get_observation()
        reward = 0  # Simplified reward
        done = False
        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.__init__()
        return self.get_observation()

    def close(self):
        p.disconnect()

    def get_object_position(self, object_name):
        for obj_data in self.objects:
            if obj_data["name"] == object_name:
                pos, _ = p.getBasePositionAndOrientation(obj_data["id"])
                return pos
        return None

    def check_task_success(self, action, obj, expected_action, expected_obj):
        if action != expected_action or obj != expected_obj:
            return False

        # Get current and initial positions
        current_pos = self.get_object_position(expected_obj)
        initial_pos = self.initial_positions.get(expected_obj)
        if current_pos is None or initial_pos is None:
            return False

        # Get box position for "place" action
        box_pos = self.get_object_position("box")

        if action == "push":
            # Check if object moved >0.1 meters in x-y plane
            xy_distance = np.linalg.norm(np.array(current_pos[:2]) - np.array(initial_pos[:2]))
            return xy_distance > 0.1

        elif action == "place":
            # Check if object is within box's bounding volume (±0.1m in x, y; z ≈ box height)
            if box_pos is None:
                return False
            return (abs(current_pos[0] - box_pos[0]) < 0.1 and
                    abs(current_pos[1] - box_pos[1]) < 0.1 and
                    abs(current_pos[2] - box_pos[2]) < 0.05)

        elif action == "lift":
            # Check if object was raised >0.1 meters
            z_diff = current_pos[2] - initial_pos[2]
            return z_diff > 0.1

        return False