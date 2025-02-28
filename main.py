import os
import glob
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from SceneGraph import SceneGraph
import threading
from queue import Queue
import cv2
import argparse
import json
import random
from multiprocessing import Pool
#from ged import compute_ged, apply_edit_operations, calculate_cost
#from ged import compute_ged, apply_edit_operations, compute_state_cost
from ged import compute_ged, convert_to_json_compatible,apply_edit_operations, generate_pddl_problem, save_pddl_problem_to_file, run_planner,read_plan_file,pddlActionsToGedOperations
import subprocess
import networkx as nx
import copy 
import re
import math
import traceback
from Robotiq import Robotiq85F
from experiment_runner import ExperimentRunner


class RoboticSystem:
    def __init__(self, scene_graph, task_queue, log_states=False):
        
        #self.client = RemoteAPIClient()

        self.client = RemoteAPIClient('localhost', 23000)
        
        self.sim = self.client.getObject('sim')
        self.simIK = self.client.require('simIK')
        self.joint_handles = []
        self.block_c_handle = None
        self.block_a_handle = None
        self.block_b_handle = None
        self.occ_obj_handle = None
        self.target_dummy = None
        self.robot_tip_handle = None
        self.robot_base_handle = None
        self.ik_environment = None
        self.ik_group = None
        self.gripper_base_handle = None
        self.table_1_handle = None
        self.table_1_position = None
        self.table_2_handle = None
        self.table_2_position = None
        self.scene_graph = scene_graph
        self.task_queue = task_queue
        self.image_counter = 0
        self.log_states = log_states
        self.log_counter = 0
        self.run_counter = 1  # Initialize run counter
        
        #self.gripper1 = Robotiq85F(self)

        self.curBox = 'Goal'
        self.prevBox = 'Goal'
        self.current_on_relationships = []

        self.anomaly_flag = False

        self.test_index = None
        # self.last_placed_object = None
        # self.last_edge_relationship = None

        self.last_action = None
        self.is_handling_anomaly = False  # Track whether we're currently handling an anomaly

        self.last_action_log = ''
        self.last_pick_direction = ''
        self.last_place_direction = ''

        self.helper = False

        self.current_anomalies = ''
        
        if log_states:
            if not os.path.exists('logs'):
                os.makedirs('logs')

    def get_current_state(self):
        state = {
            'joint_positions': [self.sim.getJointPosition(handle) for handle in self.joint_handles],
            'block_positions': {
                'Block A': self.sim.getObjectPosition(self.block_a_handle, -1),
                'Block B': self.sim.getObjectPosition(self.block_b_handle, -1),
                'Block C': self.sim.getObjectPosition(self.block_c_handle, -1),
                'Occ_Box': self.sim.getObjectPosition(self.occ_obj_handle, -1),
            },
            'goal_position': self.sim.getObjectPosition(self.table_2_handle, -1),
            'gripper_position': self.sim.getObjectPosition(self.robot_tip_handle, -1),
            'relationships': self.scene_graph.get_relationships(),  # Assuming a method to get current relationships
            'anomalies': self.current_anomalies,
            'action':  self.last_action_log,
            'pick_direction': self.last_pick_direction,
            'place_direction': self.last_place_direction,
            'changes': "",
        }
       
        return state

    def save_initial_state(self):
        # Dictionary to store initial positions and states
        self.initial_state = {
            'block_a_position': self.sim.getObjectPosition(self.block_a_handle, -1),
            'block_b_position': self.sim.getObjectPosition(self.block_b_handle, -1),
            'block_c_position': self.sim.getObjectPosition(self.block_c_handle, -1),
            'Occ_Box': self.sim.getObjectPosition(self.occ_obj_handle, -1),
            'gripper_position': self.sim.getObjectPosition(self.robot_tip_handle, -1),
            'table_1_position': self.sim.getObjectPosition(self.table_1_handle, -1),
            'table_2_position': self.sim.getObjectPosition(self.table_2_handle, -1),
            'joint_positions': [self.sim.getJointPosition(handle) for handle in self.joint_handles],
            'relationships': self.scene_graph.get_relationships()  # Assuming relationships are stored
        }
        print("Initial state saved.")
    
    def set_initial_positions(self, block_a_pos=None, block_b_pos=None, block_c_pos=None, occ_obj_pos=None):
     
        """
        Set the initial positions of blocks A, B, and C in the simulation.
        Args:
            block_a_pos: [x, y, z] coordinates for block A (optional).
            block_b_pos: [x, y, z] coordinates for block B (optional).
            block_c_pos: [x, y, z] coordinates for block C (optional).
        """
        # Set block positions if provided
        if block_a_pos is not None:
            self.sim.setObjectPosition(self.block_a_handle, -1, block_a_pos)
        if block_b_pos is not None:
            self.sim.setObjectPosition(self.block_b_handle, -1, block_b_pos)
        if block_c_pos is not None:
            self.sim.setObjectPosition(self.block_c_handle, -1, block_c_pos)
        if occ_obj_pos is not None:
            self.sim.setObjectPosition(self.occ_obj_handle, -1, occ_obj_pos)
        
        print("Initial positions set.")

    def run_planner(self):
        # Command to execute the planner
        command = [
            #'/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/pddlstream/examples/pybullet/kuka',
            'python3',
            '/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/pddlstream/examples/pybullet/kuka/run.py',
            #'--search', 'astar(lmcut())'
        ]
        
        try:
            # Run the planner and wait for it to complete
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            # Print the output from the planner if needed (stdout)
            print(result.stdout)
            
            # You can also capture stderr if there's an error
            if result.stderr:
                print(f"Error: {result.stderr}")
        
        except subprocess.CalledProcessError as e:
            print(f"Planner failed with error: {e}")
    
    def reset_to_initial_state(self):
        # Reset block positions
        self.sim.setObjectPosition(self.block_a_handle, -1, self.initial_state['block_a_position'])
        self.sim.setObjectPosition(self.block_b_handle, -1, self.initial_state['block_b_position'])
        self.sim.setObjectPosition(self.block_c_handle, -1, self.initial_state['block_c_position'])
        self.sim.setObjectPosition(self.occ_obj_handle, -1, self.initial_state['Occ_Box'])

        # Reset gripper position
        self.sim.setObjectPosition(self.robot_tip_handle, -1, self.initial_state['gripper_position'])
        
        # Reset joint positions
        for joint_handle, joint_position in zip(self.joint_handles, self.initial_state['joint_positions']):
            self.sim.setJointPosition(joint_handle, joint_position)
        
        # Restore relationships in the scene graph
        self.scene_graph.reset()  # Clear the current scene graph
        self.scene_graph.update_graph(add_edges=self.initial_state['relationships'])
        
        print("State reset to the initial saved state.")


    def clear_image_directories(self):
        for directory in ['images2', 'images']:
            files = glob.glob(os.path.join(directory, '*'))
            for f in files:
                os.remove(f)

    def connect(self):
        self.sim.startSimulation()
        print('Connected to remote API server')

    def start_simulation(self):
        self.sim.startSimulation()
        print('Simulation started')

    def stop_simulation(self):
        self.sim.stopSimulation()
        print('Simulation stopped')

    def initialize_handles(self):
        joint_names = [
            '/LBRiiwa14R820/joint1', '/LBRiiwa14R820/joint2', '/LBRiiwa14R820/joint3',
            '/LBRiiwa14R820/joint4', '/LBRiiwa14R820/joint5', '/LBRiiwa14R820/joint6','/LBRiiwa14R820/joint7'
        ]
        for joint_name in joint_names:
            handle = self.sim.getObject(joint_name)
            self.joint_handles.append(handle)
            print(f'Got handle for {joint_name}: {handle}')

        self.block_c_handle = self.sim.getObject('/Block_C')
        self.block_a_handle = self.sim.getObject('/Block_A')
        self.block_b_handle = self.sim.getObject('/Block_B')
        self.occ_obj_handle = self.sim.getObject('/Occ_Obj')
        print(f'Got handle for /Block_C: {self.block_c_handle}')
        print(f'Got handle for /Block_A: {self.block_a_handle}')
        print(f'Got handle for /Block_B: {self.block_b_handle}')
        print(f'Got handle for /Occ_Obj: {self.occ_obj_handle}')

        self.robot_base_handle = self.sim.getObject('/LBRiiwa14R820')
        print(f'Got handle for /UR10: {self.robot_base_handle}')

        self.robot_tip_handle = self.sim.getObject('/LBRiiwa14R820/connection/tip')
        print(f'Got handle for /UR10/tip: {self.robot_tip_handle}')

        # Set gripper handle based on provided information
        self.gripper_base_handle = self.sim.getObject('/LBRiiwa14R820/connection')
        print(f'Got handle for gripper base: {self.gripper_base_handle}')

        self.table_1_handle = self.sim.getObject('/T1_Dummy')
        self.table_1_position = self.sim.getObjectPosition(self.table_1_handle, -1)
        print(f'Got handle for /T1_Dummy: {self.table_1_handle}')
        print(f'Table_1_position: {self.table_1_position}')

        self.table_2_handle = self.sim.getObject('/Goal')
        self.table_2_position = self.sim.getObjectPosition(self.table_2_handle, -1)
        print(f'Got handle for /Goal: {self.table_2_handle}')
        print(f'Table_2 position: {self.table_2_position}')
    
    def turn_off_box_dynamics(self):
        self.block_c_handle = self.sim.getObject('/Block_C')
        self.block_a_handle = self.sim.getObject('/Block_A')
        self.block_b_handle = self.sim.getObject('/Block_B')
        self.occ_obj_handle = self.sim.getObject('/Occ_Obj')
        
        self.sim.setObjectInt32Parameter(self.block_a_handle, 3003, 1)
        self.sim.setObjectInt32Parameter(self.block_b_handle, 3003, 1)
        self.sim.setObjectInt32Parameter(self.block_c_handle, 3003, 1)
        self.sim.setObjectInt32Parameter(self.occ_obj_handle, 3003, 1)

        print("I turned off the dynamic of the boxes!")
    
    def turn_on_box_dynamics(self):
        self.block_c_handle = self.sim.getObject('/Block_C')
        self.block_a_handle = self.sim.getObject('/Block_A')
        self.block_b_handle = self.sim.getObject('/Block_B')
        self.occ_obj_handle = self.sim.getObject('/Occ_Obj')
        
        self.sim.setObjectInt32Parameter(self.block_a_handle, 3003, 0)
        self.sim.setObjectInt32Parameter(self.block_b_handle, 3003, 0)
        self.sim.setObjectInt32Parameter(self.block_c_handle, 3003, 0)
        self.sim.setObjectInt32Parameter(self.occ_obj_handle, 3003, 0)

        print("I turned on the dynamic of the boxes!")
       


    def create_dummy(self, position, orientation):
        dummy_handle = self.sim.createDummy(0.0001)
        self.sim.setObjectPosition(dummy_handle, -1, position)
        self.sim.setObjectOrientation(dummy_handle, -1, orientation)
        print(f'Dummy created at position: {position} with orientation: {orientation}')
        
        return dummy_handle
    
    def setup_ik(self):
        self.ik_environment = self.simIK.createEnvironment()
        self.ik_group = self.simIK.createGroup(self.ik_environment)
        self.simIK.setGroupCalculation(self.ik_environment, self.ik_group, self.simIK.method_damped_least_squares, 0.3, 99)
        
        # Debug information
        constraints = 0  # Use a default value or check self.simIK constraints
        if hasattr(self.simIK, 'constraint_pose'):
            constraints = self.simIK.constraint_pose
            print(f'Using constraint_pose: {constraints}')
        else:
            print('constraint_pose not found, using default constraints')
        
        print('Adding element from scene with:')
        print(f'ik_environment: {self.ik_environment}, type: {type(self.ik_environment)}')
        print(f'ik_group: {self.ik_group}, type: {type(self.ik_group)}')
        print(f'robot_base_handle: {self.robot_base_handle}, type: {type(self.robot_base_handle)}')
        print(f'robot_tip_handle: {self.robot_tip_handle}, type: {type(self.robot_tip_handle)}')
        print(f'target_dummy: {self.target_dummy}, type: {type(self.target_dummy)}')
        print(f'constraints: {constraints}, type: {type(constraints)}')

        try:
            self.simIK.addElementFromScene(self.ik_environment, self.ik_group, self.robot_base_handle, self.robot_tip_handle, self.target_dummy, constraints)
        except Exception as e:
            print(f"Error setting up IK: {e}")

    def move_to_position(self, target_position):
     
        if self.target_dummy is not None:
            self.sim.setObjectPosition(self.target_dummy, -1, target_position)
            self.setup_ik()
            result, reason, precision = self.simIK.handleGroup(self.ik_environment, self.ik_group, {'syncWorlds': 'True'})
            tries = 0
            while True:
                res = self.get_distance(self.robot_tip_handle, self.target_dummy)
                if res < 0.01 or tries > 1000: #0.01
                    break
                # if tries > 500:
                #     time.sleep(2)
                #     self.scene_graph.reset()  # Reset the scene graph anomaly flag
                #     print("I failed to reach pose so I will just restart.")
                #     time.sleep(5)  # Wait a bit to see the action
                #     self.reset_simulation()
                tries+= 1
        
            if result == 1:
                print('IK calculation succeeded')
                return True
            else:
                print(f'Failed to handle IK group, reason: {reason}, precision: {precision}, res: {res}, tries: {tries}')
                return False
        else:
            print("Error: Target dummy handle is None. Cannot move to position.")
            return False
            
    def get_distance(self, handle1, handle2):
        position1 = self.sim.getObjectPosition(handle1, -1)
        position2 = self.sim.getObjectPosition(handle2, -1)
        # Calculate the Euclidean distance in 3D space (x, y, z)
        return sum((p1 - p2) ** 2 for p1, p2 in zip(position1, position2)) ** 0.5


    def store_last_action(self, action_name, *args, **kwargs):
        """Stores the last action executed for potential re-execution."""
        self.last_action = (action_name, args, kwargs)
    
    def execute_last_action(self):
        """Re-executes the last stored action."""
        if self.last_action is not None:
            action_name, args, kwargs = self.last_action
            action_method = getattr(self, action_name, None)
            if action_method:
                print(f"Re-executing last action: {action_name} with args {args} and kwargs {kwargs}")
                action_method(*args, **kwargs)
            else:
                print(f"Error: No method named {action_name} found.")
        else:
            print("No last action to execute.")
    
    def is_holding(self, gripper_position, block_position, threshold=0.1):
        distance = sum((g - b) ** 2 for g, b in zip(gripper_position, block_position)) ** 0.5
        #print(f"what is distance? = {distance}")
        if distance < threshold:
            return True
        return False
    

    def is_on(self, position_u, position_v, z_threshold=0.12, xy_threshold=0.06):
       # Ensure object u is higher in z-axis than object v
        if position_u[2] > position_v[2]:
            # Check if the bottom of u is close to the top of v with a smaller threshold
            if abs(position_u[2] - position_v[2]) < z_threshold:
                # Ensure horizontal alignment
                if abs(position_u[0] - position_v[0]) < xy_threshold and abs(position_u[1] - position_v[1]) < xy_threshold:
                    return True
        return False
    

    def is_on_table(self, block_position, table_position, z_threshold=0.10, xy_threshold=0.2):
        # Similar logic as is_on(), but for the table
        return self.is_on(block_position, table_position, z_threshold, xy_threshold)

    
    def update_scene_graph(self):
        '''
        Need to comeback here to figureout how to include the occluded object. 
        '''
        # Create a consistent name mapping for blocks and boxes
        name_mapping = {
            'Block A': 'Box A',
            'Block B': 'Box B',
            'Block C': 'Box C',
            'Occ_Box': 'Occ_Box'
        }

        # Step 1: Extract the current state from the simulation
        current_state = self.get_current_state()

        # Get block positions from the current state
        block_positions = current_state['block_positions']
        gripper_position = current_state['gripper_position']

        # Initialize lists for adding and removing edges
        add_edges = []
        remove_edges = []

        # Step 2: Determine relationships (on, holding)

        # Remove old "on" relationships for blocks that are now being held
        for block_name, block_position in block_positions.items():
            if self.is_holding(gripper_position, block_position):
                # Gripper is holding this block, so remove any "on" relationship it previously had
                for target_block in block_positions:
                    remove_edges.append((name_mapping[block_name], name_mapping.get(target_block, ''), 'on'))
                # Also remove any "on table" relationships
                remove_edges.append((name_mapping[block_name], 'Table 1', 'on'))
                remove_edges.append((name_mapping[block_name], 'Goal', 'on'))

                # Add the new holding relationship
                add_edges.append(('Gripper', name_mapping[block_name], 'holding'))

        # Check if blocks are on other blocks or tables
        for block_name_u, position_u in block_positions.items():
            for block_name_v, position_v in block_positions.items():
                if block_name_u != block_name_v:
                    if self.is_on(position_u, position_v):
                        # Block u is on block v, remove any "holding" relationship and add the "on" relationship
                        add_edges.append((name_mapping[block_name_u], name_mapping[block_name_v], 'on'))
                    remove_edges.append(('Gripper', name_mapping[block_name_u], 'holding'))

            # Check if block is on a table
            if self.is_on_table(position_u, self.table_1_position):
                add_edges.append((name_mapping[block_name_u], 'Table 1', 'on'))
            elif self.is_on_table(position_u, self.table_2_position):
                add_edges.append((name_mapping[block_name_u], 'Goal', 'on'))


        allEdges = self.scene_graph.get_relationships()
        # Step 3: Update the scene graph by adding or removing edges
        self.scene_graph.update_graph(
            add_edges=add_edges,
            remove_edges=allEdges
        )

        # Capture and display the updated scene graph
        #filename = f'images2/scene_pick_{int(time.time())}.png'
        #self.scene_graph.capture_image(filename)
        #self.task_queue.put(lambda: self.display_image(filename))

        time.sleep(0.2)
        print("Scene graph updated with real-time data.")

    def push(self, box_handle, target_position, threshold=0.01, step=0.005):
        """
        Pushes the box towards the target position until the position is within the threshold.
        
        :param box_handle: Handle of the box to be pushed.
        :param target_position: The target [x, y, z] position to push the box towards.
        :param threshold: The acceptable distance from the target position.
        :param step: The distance to move the box in each push iteration.
        """

        while True:
            # Get the current position of the box
            current_position = self.sim.getObjectPosition(box_handle, -1)
            
            # Calculate the difference between the current position and target position
            offset_x = target_position[0] - current_position[0]
            offset_y = target_position[1] - current_position[1]
            
            # Check if the box is within the acceptable threshold
            if abs(offset_x) <= threshold and abs(offset_y) <= threshold:
                print("Box is aligned correctly.")
                break
            
            # Determine the direction of the push
            direction_x = step if offset_x > 0 else -step
            direction_y = step if offset_y > 0 else -step
            
            # Calculate the new position
            new_position = [current_position[0] + direction_x,
                            current_position[1] + direction_y,
                            current_position[2]]  # Z remains the same during the push
            
            # Move the box to the new position
            self.sim.setObjectPosition(box_handle, -1, new_position)
            
            # Small delay to simulate the pushing process
            time.sleep(0.1)
            
        print("Finished pushing the box.")

    def pick(self, object_handle):
        #time.sleep(0.5)
        self.update_scene_graph()
        #time.sleep(0.5)
       
        if object_handle and self.gripper_base_handle:
            self.store_last_action('pick', object_handle)
            # Simulate picking up the object
            self.sim.setObjectInt32Parameter(object_handle, 3003, 1)
            #time.sleep(0.3)
            self.sim.resetDynamicObject(object_handle)
            #time.sleep(0.3)
            self.sim.setObjectParent(object_handle, self.gripper_base_handle, True)
            #time.sleep(0.3)
            print(f'Picked up the object (simulated)')
            #time.sleep(0.5)
            self.update_scene_graph()
            #time.sleep(0.5)
        else:
            print('Error: Object handle or gripper handle is not valid.')
        time.sleep(0.5)  # Delay after picking

    
    def new_pick(self, object_handle):

        dummy_orientation = [-1.3904, -2, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
        
        block_position = self.sim.getObjectPosition(object_handle, -1)

        dummy_handle = self.create_dummy(block_position, dummy_orientation)
        if dummy_handle is not None:
            
            self.target_dummy = dummy_handle
            #self.move_to_position([self.table_1_position[0]+0.035, self.table_1_position[1], self.table_1_position[2]+0.7])
            dummy_handle = self.create_dummy(block_position, dummy_orientation)
            self.target_dummy = dummy_handle    


            print("I am inside pick, I am moving above the object")
            move_checker = self.move_to_position([block_position[0], block_position[1], block_position[2]+0.1])
            if move_checker == False:
                return False
            time.sleep(0.5)
            print("I am inside pick, I am moving closer the object")
            move_checker = self.move_to_position([block_position[0], block_position[1], block_position[2]+0.04])
            if move_checker == False:
                return False
            time.sleep(0.5)

        self.update_scene_graph()
        time.sleep(0.5)
       
        if object_handle and self.gripper_base_handle:
            self.store_last_action('pick', object_handle)
            # Simulate picking up the object
            self.sim.setObjectInt32Parameter(object_handle, 3003, 1)
            #time.sleep(0.3)
            self.sim.resetDynamicObject(object_handle)
            #time.sleep(0.3)
            self.sim.setObjectParent(object_handle, self.gripper_base_handle, True)
            #time.sleep(0.3)
            print(f'Picked up the object (simulated)')
            #time.sleep(0.5)
            self.update_scene_graph()
            #time.sleep(0.5)
            move_checker = self.move_to_position([block_position[0], block_position[1], block_position[2]+0.3])
            if move_checker == False:
                return False
            time.sleep(0.5)
          

        else:
            print('Error: Object handle or gripper handle is not valid.')
        time.sleep(0.5)  # Delay after picking


    def new_place(self, holding_handle, target_hadle):

        dummy_orientation = [-1.3904, -2, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
        
        
        if target_hadle == "temp":

            block_position = self.sim.getObjectPosition(self.block_a_handle, -1)
            
            dummy_handle = self.create_dummy(block_position, dummy_orientation)
            if dummy_handle is not None:
                
                self.target_dummy = dummy_handle
                dummy_handle = self.create_dummy(block_position, dummy_orientation)
                self.target_dummy = dummy_handle
                print("I am inside place, I am moving above the temp location")
                self.move_to_position([-0.4, 0.4, 0.5])
                time.sleep(2)

            self.update_scene_graph()
            time.sleep(0.5)


        else:

            block_position = self.sim.getObjectPosition(target_hadle, -1)

            dummy_handle = self.create_dummy(block_position, dummy_orientation)
            if dummy_handle is not None:
                
                self.target_dummy = dummy_handle
                #self.move_to_position([self.table_1_position[0]+0.035, self.table_1_position[1], self.table_1_position[2]+0.7])
                

                dummy_handle = self.create_dummy(block_position, dummy_orientation)
                self.target_dummy = dummy_handle
                print("I am inside place, I am moving above the target location")
                move_checker = self.move_to_position([block_position[0], block_position[1], block_position[2]+0.4])
                if move_checker == False:
                    return False
                time.sleep(1.5)

                x_offset = random.uniform(*(-0.05,0.05))
                y_offset = random.uniform(*(-0.05,0.05))
                z_offset = random.uniform(*(0.2,0.3))

                print(f"I am about to place and the x_offset = {x_offset} | y_offset = {y_offset} | z_offset = {z_offset}")
                #Now need to randomize the x,y,z for when I try to place the object 
                if self.helper == False:
                    move_checker = self.move_to_position([block_position[0]+x_offset,block_position[1]+y_offset, block_position[2]+z_offset])
                else:
                    move_checker = self.move_to_position([block_position[0],block_position[1], block_position[2]+z_offset])
                if move_checker == False:
                    return False
                time.sleep(1)

            self.update_scene_graph()
            time.sleep(0.5)


        if holding_handle and self.gripper_base_handle:
            self.store_last_action('place', holding_handle)
            time.sleep(1)
            # Simulate placing the object
            self.sim.setObjectParent(holding_handle, -1, True)
            time.sleep(0.3)
            self.sim.setObjectInt32Parameter(holding_handle, 3003, 0)
            time.sleep(1.0)
            #Below make sure that dynamics are turned off after placing to make sure it dont slide down afterwards...
            self.sim.setObjectInt32Parameter(holding_handle, 3003, 0)
            print(f'Placed the object (simulated)')

            time.sleep(0.5)
            self.update_scene_graph()
            time.sleep(0.5)
            print("Inside Place action")

            move_checker = self.move_to_position([block_position[0], block_position[1], block_position[2]+0.4])
            if move_checker == False:
                return False



        else:
            print('Error: Object handle or gripper handle is not valid.')
        time.sleep(0.5)  # Delay after placing

    def place(self, object_handle):
        if object_handle and self.gripper_base_handle:
            self.store_last_action('place', object_handle)
            time.sleep(1)
            # Simulate placing the object
            self.sim.setObjectParent(object_handle, -1, True)
            time.sleep(0.3)
            self.sim.setObjectInt32Parameter(object_handle, 3003, 0)
            time.sleep(1.0)
            #Below make sure that dynamics are turned off after placing to make sure it dont slide down afterwards...
            self.sim.setObjectInt32Parameter(object_handle, 3003, 1)
            print(f'Placed the object (simulated)')

            # target_node = 'Goal'
            #  # Check for the most recent placed box
            # for source, target, data in self.scene_graph.graph.edges(data=True):
            #     if target == target_node and data['label'] == 'on':
            #         target_node = source
            # node_name = self.scene_graph.handle_to_node[object_handle]
            
            # if self.curBox == None:
            #     self.curBox = node_name
            # else:
            #     self.prevBox = self.curBox
            #     self.curBox = node_name
            
            # self.current_on_relationships.append((self.curBox, self.prevBox))

            # print(f"I am here... {self.current_on_relationships}")
            
            
            # self.scene_graph.current_on_relationships = self.current_on_relationships
            # time.sleep(2)
            # self.scene_graph.object_postition = self.get_current_state()
            
            # anomalies = self.scene_graph.detect_anomalies()
            # print(f"Is there really an anomaly? {anomalies}")
            # self.current_anomalies = anomalies
            
            #time.sleep(0.5)
            #self.update_scene_graph()
            #time.sleep(0.5)

            # if anomalies:
            #      self.anomaly_flag = True

            # #print("Inside place", anomalies, len(anomalies), "Overturn" in anomalies[0])
            # if not self.log_states:
            #     if anomalies:
            #         self.store_last_action('place', object_handle)
            #         print("\n".join(anomalies))
            #         if len(anomalies): # May need to do if len == 2 and overturn inside etc. 
            #             time.sleep(0.5)
            #             self.update_scene_graph()
            #             time.sleep(0.5)

            #         if not self.is_handling_anomaly:
            #             self.is_handling_anomaly = True
            #             self.anomaly_handling(anomalies)  # Call the anomaly handling function
            #             self.is_handling_anomaly = False

            #time.sleep(0.5)
            self.update_scene_graph()
            #time.sleep(0.5)
            print("Inside Place action")

        else:
            print('Error: Object handle or gripper handle is not valid.')
        time.sleep(0.5)  # Delay after placing


    def move_and_place_block(self, block_handle, block_position, table_position, lift_range=(0.6, 0.6), place_range_1=(0.7, 0.7), place_range_2=(0.6, 0.6)):
        # for _test in range(2):

        #block_position = self.sim.getObjectPosition(self.block_c_handle, -1)

        lift_height = random.uniform(*lift_range)
        place_height_1 = random.uniform(*place_range_1)
        place_height_2 = random.uniform(*place_range_2)

        print(f'Moving block from {block_position} to table at {table_position}')
        #dummy_orientation = [-1.6904, -1.1574, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°

        dummy_orientation = [-1.7904, -7, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°

        
        # Update block position to get the latest position
        self.scene_graph.update_positions()
        block_position = self.sim.getObjectPosition(block_handle, -1)

        # Generate a random value between 0, 1, or 2
        #random_value = random.randint(0, 2)
        random_value = 0

        #Create dummy above the block and move to it

        #self.gripper1.openGripper()
        #time.sleep(1)
    
        #block_position[2] += 0.05
       
        dummy_handle = self.create_dummy(block_position, dummy_orientation)
        if dummy_handle is not None:
            
            self.target_dummy = dummy_handle
            self.move_to_position([self.table_1_position[0]+0.035, self.table_1_position[1], self.table_1_position[2]+0.7])
            if random_value == 0: #Picking above (top)
                dummy_orientation = [-1.7904, -7, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
                block_position[2] += 0.08  # Move above the block
            elif random_value == 1: #Picking left 
                #dummy_orientation_right = [-1.6904, 0, -3.1416]  # Facing right side, 180 degrees yaw rotation
                dummy_orientation = [-1.6904, 0, -0.9]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
                block_position[0] += 0.01  # Move to the left of the block
            elif random_value == 2: #Picking right
                dummy_orientation = [-1.6904, 0, -2.5]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
                block_position[0] -= 0.01  # Move to the right of the block
    
            
            dummy_handle = self.create_dummy(block_position, dummy_orientation)
            self.target_dummy = dummy_handle
            
            self.move_to_position(block_position)
            time.sleep(0.3)

            self.update_scene_graph()
            if self.log_states:
                self.log_state(self.anomaly_flag)
                time.sleep(0.3)
            else:
                self.log_counter+=1

            self.last_action_log = 'pick'
            
            if random_value == 0:
                self.last_pick_direction = 'pick_top'
            elif random_value == 1:
                self.last_pick_direction = 'pick_left'
            else:
                self.last_pick_direction = 'pick_right'
            #self.gripper1.closeGripper(block_handle)
            #time.sleep(1)
            #isGripperClosed = self.gripper1.closeGripper(block_handle)
            #time.sleep(2)
            # for joint in self.joint_handles:
            #     self.sim.setJointTargetVelocity(joint, 0)
            # Wait before picking
            time.sleep(0.2)  # 1.5-second delay before picking


            self.pick(block_handle)
            time.sleep(0.2)
            
            self.update_scene_graph()
            if self.log_states:
                self.log_state(self.anomaly_flag)
                time.sleep(0.3)
                self.last_action_log = ''
                self.last_pick_direction = ''
            else:
                self.log_counter+=1

            # Lift the block
            lift_position = block_position[:]
            # lift_position[2] += lift_height
            lift_position[2] += 0.5
            self.move_to_position(lift_position)
            time.sleep(0.3)

            time.sleep(0.3)
            self.update_scene_graph()
            #time.sleep(1)

            # Remove any "on" relationships for the current block
            node_name = self.scene_graph.handle_to_node[block_handle]
            remove_edges = []

            self.update_scene_graph()
            print("I am here 1")
            # Move to table position 1
            table_position_with_height = table_position[:]
            table_position_with_height[2] += 0.5
            #self.move_to_position(table_position_with_height)
            print(f"I am here 2, table_post = {table_position_with_height}")
            # Check distance before placing
            self.move_to_position(table_position_with_height)
            time.sleep(1)
            print("I am here 3")
            time.sleep(0.6)
            
            # Move to table position 2
            table_position_with_height = table_position[:]
            table_position_with_height[2] += place_height_2


            if random_value == 1:
                table_position_with_height[0] += 0.05
            
            if random_value == 2:
                table_position_with_height[0] -= 0.05

            # if _test == 0:
            #     _offset = 0.03
            # else:
            #     _offset = 0.0
            _offset = 0.04
            #table_position_with_height[0] = table_position_with_height[0] + _offset
            table_position_with_height[0] = random.uniform(*(table_position_with_height[0]-_offset,table_position_with_height[0]+_offset))
            table_position_with_height[1] = random.uniform(*(table_position_with_height[1]-_offset,table_position_with_height[1]+_offset))
            
            #self.move_to_position(table_position_with_height)
            print("I am here 4")
            # Check distance before placing
            self.move_to_position(table_position_with_height)
         
            time.sleep(0.5)
            print("I am here 5")
            self.update_scene_graph()
            if self.log_states:
                self.log_state(self.anomaly_flag)
                time.sleep(0.3)
            else:
                self.log_counter+=1
            
            self.last_action_log = 'place'
            if random_value == 0:
                self.last_place_direction = 'place_top'
            elif random_value == 1:
                self.last_place_direction = 'place_left'
            else:
                self.last_place_direction = 'place_right'

          
            self.place(block_handle)
            time.sleep(0.5)
            print("I am here 6")
            self.update_scene_graph()
            if self.log_states:
                time.sleep(0.3)
                self.log_state(self.anomaly_flag)
                time.sleep(0.3)
                self.last_action_log = ''
                self.last_place_direction = ''
            else:
                self.log_counter+=1
            print("I am here 7")
        else:
            print('Failed to create dummy above block')
            exit()

    def grasp_and_move(self):
        # Get positions

        self.set_initial_positions(
            block_a_pos=[-0.70, 0.4, 0.20],
            block_b_pos=[-0.72, 0.4, 0.33],
            block_c_pos=[-0.7, 0.4, 0.45]
        )

        time.sleep(1)
        self.turn_on_box_dynamics()
        time.sleep(2)
        self.turn_off_box_dynamics()
        time.sleep(1)



        block_c_position = self.sim.getObjectPosition(self.block_c_handle, -1)
        block_b_position = self.sim.getObjectPosition(self.block_b_handle, -1)
        block_a_position = self.sim.getObjectPosition(self.block_a_handle, -1)
    
        #Handling M+M
        
        #These two actions will just move it to the side
        self.new_pick(self.block_c_handle)
        self.new_place(self.block_c_handle, 'temp')

        #Now need to address box b before placing box c. 
        #(Make this happen twice for now then the final just once...)
        for _ in range(5):
            self.new_pick(self.block_b_handle)
            self.new_place(self.block_b_handle,self.block_a_handle)
        
        self.helper = True
        self.new_pick(self.block_b_handle)
        self.new_place(self.block_b_handle,self.block_a_handle)

        self.helper = False
        for _ in range(4):
            self.new_pick(self.block_c_handle)
            self.new_place(self.block_c_handle,self.block_b_handle)

        self.helper = True
        self.new_pick(self.block_c_handle)
        self.new_place(self.block_c_handle,self.block_b_handle)

        time.sleep(100)

        self.update_scene_graph()
        self.scene_graph.current_on_relationships = self.scene_graph.get_relationships()
        time.sleep(2)
        self.scene_graph.object_postition = self.get_current_state()
        
        time.sleep(3)
        self.turn_on_box_dynamics()
        time.sleep(2)

        anomalies = self.scene_graph.detect_anomalies()
        print(self.scene_graph.get_relationships())
        print(f"Is there really an anomaly? {anomalies}")
        self.current_anomalies = anomalies

        if anomalies:
                self.anomaly_flag = True
        
        #print("Inside place", anomalies, len(anomalies), "Overturn" in anomalies[0])
        if not self.log_states:
            if anomalies:
                print("\n".join(anomalies))
                if len(anomalies): # May need to do if len == 2 and overturn inside etc. 
                    time.sleep(0.5)
                    self.update_scene_graph()
                    time.sleep(0.5)

                if not self.is_handling_anomaly:
                    self.is_handling_anomaly = True
                    self.anomaly_handling(anomalies)  # Call the anomaly handling function
                    self.is_handling_anomaly = False

        # Move Block C
        self.move_and_place_block(self.block_c_handle, block_c_position, self.table_2_position)

        # Move Block B
        self.move_and_place_block(self.block_b_handle, block_b_position, self.table_2_position)
        
        # Move Block A
        self.move_and_place_block(self.block_a_handle, block_a_position, self.table_2_position)

        self.image_counter += 1
        #time.sleep(100)
    
    # A function to extract the numerical part of the log file name
    def extract_log_index(self, filename):
        match = re.search(r'Log_State_(\d+)_Run_(\d+)\.json', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return float('inf'), float('inf')  # If the pattern does not match, push it to the end
    
    def anomaly_handling(self, anomalies):
        print("Handling anomaly...")

        print("I am about to attempt the anomaly handling")

        # Retrieve current state
        current_state = self.get_current_state()
        current_state_json = convert_to_json_compatible(current_state)
        #Save the current state
        self.log_current_state()

        #Run the bayesian network to get the goal state
        #(TODO after, for now can assume this is fine, need to retrain BN actually)

        #Run PDDLStream with correct intial predicates/start and goal positions
        #(Just run the PDDLStream where we assume it has already taken in the inputs)
        #self.run_planner()

        initial_state_file = '/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/_data/my_plan.json'
        with open(initial_state_file, 'r') as f:
            plan_data = json.load(f)
        
        print(plan_data)
        time.sleep(2)

        # dummy_orientation = [-1.7904, 5, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
        dummy_orientation = [-1.3904, -1, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
       
        dummy_handle = self.create_dummy(self.sim.getObjectPosition(self.block_a_handle, -1), dummy_orientation)
        self.target_dummy = dummy_handle
        #Moving to a place where I can perform pick later
        self.move_to_position([self.table_1_position[0]+0.035, self.table_1_position[1], self.table_1_position[2]+0.7])
        print("I have moved to the location prior to perfoming recovery")
        time.sleep(1)

        # Loop through the plan
        for step in plan_data:
         
            action = step["action"]
            details = step["details"]

            # Process 'pick' actions
            if action == "pick":
                # Extract pose and grasp
             
                pose = details["pose"][0]  # The [x, y, z] part of the pose
                grasp = details["grasp"][0]  # The [x, y, z] part of the grasp
                object = details['object']

                block = None
                if int(object) == 4:
                    block = self.block_a_handle
                if int(object) == 5:
                    block = self.block_b_handle
                if int(object) == 6:
                    block = self.block_c_handle

                # Debug: Print the pose and grasp
                print(f"Pick action: Object={details['object']}, Pose={pose}, Grasp={grasp}")

                # Example usage of your move_to_position function in CoppeliaSim
                # Move to the object's pose

                # dummy_orientation = [-1.7904, -7, -1.6216]  # Euler angles: alpha=-96.853°, beta=-66.326°, gamma=-92.923°
                # dummy_handle = self.create_dummy([-pose[0], pose[1], pose[2]+0.45], dummy_orientation)
                # time.sleep(5)
               # self.target_dummy = dummy_handle
                # print(f"Moving to position prior to place")
                # self.move_to_position([-0.5, 0.4, 0.6])
                # time.sleep(1)
                
                print(f"Moving to pose...{pose}")
                #self.move_to_position([-pose[0], pose[1], pose[2]+0.31])  # Replace 'self' with your class instance or context (2 boxes)
                #self.move_to_position([-pose[0], pose[1], pose[2]-0.01])  # Replace 'self' with your class instance or context (3 boxes)
                pos = self.sim.getObjectPosition(block, -1)
                self.move_to_position([pos[0], pos[1], pos[2]+0.07])
                print("I have now moved to the location to pick")
                time.sleep(2)
                self.pick(block)
                print("I have picked the block")
                time.sleep(2)
                #self.move_to_position([-pose[0], pose[1], pose[2]+0.50]) 
                self.move_to_position([pos[0], pos[1], 0.55])
                print("Pick completed, now moving up a bit")
                time.sleep(2)

            #Process 'place' action
            if action == "place":

                '''
                Figure out why it does not move to place location? it should indeed
                be different to the pick placement?
                '''

                pose = details["pose"][0]  # The [x, y, z] part of the pose
                grasp = details["grasp"][0]  # The [x, y, z] part of the grasp


                # print(f"Moving to position prior to place")
                # self.move_to_position([-0.5, 0.3, 0.6])
                # time.sleep(1)

                # Debug: Print the pose and grasp
                print(f"Place action: Object={details['object']}, Pose={pose}, Grasp={grasp}")

                # Example usage of your move_to_position function in CoppeliaSim
                # Move to the object's pose
                print(f"Moving to pose...{pose}")
                #self.move_to_position([-pose[0], pose[1], pose[2]+0.55])  # Replace 'self' with your class instance or context
                self.move_to_position([-pose[0], pose[1], pose[2]+0.1])  # Replace 'self' with your class instance or context
                print("I have now moved to the location to place")
                time.sleep(2)
                self.place(block)
                print("I have placed the block")
                time.sleep(2)
                # #self.move_to_position([-pose[0], pose[1], pose[2]+0.70]) 
                # self.move_to_position([-pose[0], pose[1], pose[2]+0.60]) 
                # print("Place completed, now moving up a bit")
                # time.sleep(2)
                print(f"Moving to position prior to place")
                self.move_to_position([-0.2, 0.3, 0.6])
                time.sleep(2)
                
                # # Move to the grasp pose (if needed)
                # print("Moving to grasp...")
                # self.move_to_position(grasp)

        #Using the result from the PDDLStream to solve the anomaly


        #Last check to see if the anomaly is still there. 
        exit()
        
        print("Finished anomaly handling.")
        print("Exiting anomaly handling.")

    def display_image(self, filename):
        # Check if the image file exists and is valid
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            img = cv2.imread(filename)
            if img is not None and img.size > 0:
                resized_img = cv2.resize(img, (1000, 600))  # Resize to 1000x600 pixels
                cv2.imshow('Scene Graph', resized_img)
                cv2.waitKey(1000)  # Display the image for 1 second
            else:
                print(f"Warning: The image {filename} could not be read properly.")
        else:
            print(f"Warning: The image file {filename} does not exist or is empty.")

    def reset_simulation(self):

        self.sim.stopSimulation()
        time.sleep(2)  # Ensure the simulation is fully stopped
        print('Simulation stopped for reset')
        #self.initialize_boxes()
        time.sleep(0.5)
        self.sim.startSimulation()
        print('Simulation restarted')
        #time.sleep(5)  # Allow some time for the scene to stabilize
        time.sleep(2)  # Allow some time for the scene to stabilize

        # Reset other variables
        self.log_counter = 0
        self.curBox = 'Goal'
        self.prevBox = 'Goal'
        self.current_on_relationships = []
        self.anomaly_flag = False
        self.test_index = None
        self.last_action = None
        self.is_handling_anomaly = False
        self.current_anomalies = ''
        self.last_action_log  = None
        print("I am about to reset scene graph")
        self.scene_graph.reset()  # Reset the scene graph anomaly flag
        print("I have resetted the scene graph")
        time.sleep(0.5)

    def log_state(self, flag):
        state_data = self.get_current_state()
        with open(f'logs/kuka_logs/Log_State_{self.log_counter+1}_Run_{self.run_counter}.json', 'w') as f:
                json.dump(state_data, f)
        self.log_counter += 1
    
    def log_current_state(self, changes):
        state_data = self.get_current_state()
        state_data['changes'] = changes
        with open(f'logs/current_state/current_state.json', 'w') as f:
                json.dump(state_data, f)
        self.log_counter += 1

        # if flag == False:
        #     with open(f'logs_success/Log_State_{self.log_counter+1}_Run_{self.run_counter}.json', 'w') as f:
        #         json.dump(state_data, f)
        #     self.log_counter += 1
        # else:
        #     with open(f'logs_fail/Log_State_{self.log_counter+1}_Run_{self.run_counter}.json', 'w') as f:
        #         json.dump(state_data, f)
        #     self.log_counter += 1
        

# Function to handle each simulation run
def run_simulation(run_id, log_dir, scene_file, headless=False):
    # Create the unique log directory for this simulation run
    unique_log_dir = f'{log_dir}/run_{run_id}'
    if not os.path.exists(unique_log_dir):
        os.makedirs(unique_log_dir)

    # Use the full path to CoppeliaSim executable
    coppeliaSim_path = '/home/yazz/Downloads/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu22_04/coppeliaSim.sh'

    # Run CoppeliaSim in headless mode with the provided scene file
    if headless:
        # Full command to run CoppeliaSim in headless mode
        cmd = [coppeliaSim_path, '-h', scene_file]
        print(f"Running headless CoppeliaSim for run {run_id} with command: {' '.join(cmd)}")
    else:
        # If not headless, we run the main.py script in regular mode (for testing)
        cmd = ['python3', 'main.py', '--log', '--log_dir', unique_log_dir]

    # Run the command as a subprocess
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='Enable state logging')
    args = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--log', action='store_true', help='Enable state logging')
    # parser.add_argument('--parallel', type=int, help='Number of parallel simulations to run')
    # parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs')
    # parser.add_argument('--headless', action='store_true', help='Run CoppeliaSim in headless mode')
    # parser.add_argument('--scene_file', type=str, required=True, help='Path to CoppeliaSim scene file')

    # args = parser.parse_args()

    # # If --parallel argument is passed, run parallel simulations
    # if args.parallel:
    #     # Create the main log directory if it doesn't exist
    #     if not os.path.exists(args.log_dir):
    #         os.makedirs(args.log_dir)

    #     # Run parallel simulations using multiprocessing Pool
    #     with Pool(processes=args.parallel) as pool:
    #         pool.starmap(run_simulation, [(i, args.log_dir, args.scene_file, args.headless) for i in range(args.parallel)])
        
    #     return  # Exit after running parallel simulations

    task_queue = Queue()
    
    scene_graph = SceneGraph(task_queue)

    robotic_system = RoboticSystem(scene_graph, task_queue, log_states=args.log)
    
    # Start the scene graph thread
    scene_graph_thread = threading.Thread(target=scene_graph.run, daemon=True)
    scene_graph_thread.start()

    # Start the main thread
    task_thread = threading.Thread(target=process_tasks, args=(task_queue,), daemon=True)
    task_thread.start()

    robotic_system.connect()
    robotic_system.start_simulation()
    robotic_system.initialize_handles()

    robotic_system.turn_off_box_dynamics()

    #Misalignment on B example, when A,B Stacked
    # robotic_system.set_initial_positions(
    # block_a_pos=[-0.7, 0.4, 0.17],
    # block_b_pos=[-0.73, 0.4, 0.28],
    # )



    # #Overtun on B example, when A,B Stacked
    # robotic_system.set_initial_positions(
    # block_a_pos=[-0.7, 0.4, 0.17],
    # block_b_pos=[-0.75, 0.4, 0.28],
    # )



    # #Misalignement on B example when A,B,C Stacked
    # robotic_system.set_initial_positions(
    # block_a_pos=[-0.7, 0.4, 0.17],
    # block_b_pos=[-0.73, 0.4, 0.28],
    # block_c_pos=[-0.7, 0.4, 0.39],
    # )



    # #Overtun and Misalignement on B,C example when A,B,C Stacked
    # robotic_system.set_initial_positions(
    # block_a_pos=[-0.7, 0.4, 0.17],
    # block_b_pos=[-0.73, 0.4, 0.28],
    # block_c_pos=[-0.64, 0.4, 0.39],
    # )

    # #Overtun on B,C example when A,B,C Stacked
    # robotic_system.set_initial_positions(
    # block_a_pos=[-0.7, 0.4, 0.17],
    # block_b_pos=[-0.75, 0.4, 0.28],
    # block_c_pos=[-0.68, 0.4, 0.39],
    # )

    # time.sleep(1)
    # robotic_system.turn_on_box_dynamics()
    # time.sleep(100)
    
    #I need to start, take relationship the turn on dynamics.. 

    #robotic_system.save_initial_state()
    # robotic_system.generateURDF()
    # exit()

    #Misalignment on B example, when A,B Stacked
    # robotic_system.set_initial_positions(
    # block_a_pos=[-0.7, 0.4, 0.17],
    # block_b_pos=[-0.73, 0.4, 0.28],
    # occ_obj_pos=[-0.7, 0.6, 0.25]
    # )

    ########## Experiement Pipeline ##########
    #Initialize experiment runner
    experiment_runner = ExperimentRunner(robotic_system, num_trials=50)
    #Run experiments for both methods
    #["M_B", "O_B", "M_A_B_M_B_C", "M_B_O_C", "O_B_O_C"] 
    for anomaly in ["M_B_O_C"]:
        #experiment_runner.run_experiment(anomaly, "nominal_1")
        #experiment_runner.run_experiment(anomaly, "nominal_2")
        experiment_runner.run_experiment(anomaly, "BN_PDDLStream")


    ########## Result Pipeline ##########

    # M_B = {}
    # O_B = {}
    # M_A_B_M_B_C = {}
    # M_B_O_C = {}
    # O_B_O_C = {}

    # initial_state_file = '/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/_data/experiment_results_new.json'
    # with open(initial_state_file, 'r') as f:
    #     result_data_1 = json.load(f)

    # initial_state_file = '/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/_data/experiment_results_new_new.json'
    # with open(initial_state_file, 'r') as f:
    #     result_data_2 = json.load(f)



    # print(f"This is result data 1 {result_data_1} \n")

    # print(f"This is result data 2 {result_data_2} \n")

    # exit()

    # import json
    # from collections import defaultdict
    # import pandas as pd
    # import tools
    # #import ace_tools
    # #import ace_tools as tools

    # # Initialize data storage for anomalies
    # anomalies = {
    #     "M_B": defaultdict(list),
    #     "O_B": defaultdict(list),
    #     "M_A_B_M_B_C": defaultdict(list),
    #     "M_B_O_C": defaultdict(list),
    #     "O_B_O_C": defaultdict(list),
    # }

    # # Load both JSON files
    # json_files = [
    #     "/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/_data/experiment_results.json",
    #     "/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/_data/experiment_results_1.json",
    #     "/home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/_data/experiment_results_2.json"
    # ]

    # # Define the max number of trials per anomaly-method pair
    # MAX_TRIALS = 50  # Assuming each anomaly-method has 50 total trials

    # for file in json_files:
    #     with open(file, 'r') as f:
    #         result_data = json.load(f)
    #         for trial in result_data:
    #             anomaly_type = trial["anomaly_type"]
    #             method = trial["method"]
                
    #             if anomaly_type in anomalies:
    #                 anomalies[anomaly_type][method].append(trial)

    # # Compute statistics
    # summary = {}

    # for anomaly, methods in anomalies.items():
    #     summary[anomaly] = {}
    #     for method, trials in methods.items():
    #         total_runs = len(trials)  # Total trials recorded
    #         successful_trials = [t for t in trials if t["success"] == 1]  # Filter successful ones
    #         success_count = len(successful_trials)  # Count successful trials

    #         # Success Rate Calculation
    #         success_rate = (success_count / MAX_TRIALS) * 100  # Divide by max trials

    #         # Compute NA and TT only for successful trials
    #         if success_count > 0:
    #             avg_actions = sum(t["actions_used"] for t in successful_trials) / success_count
    #             avg_time = sum(t["time_taken"] for t in successful_trials) / success_count
    #         else:
    #             avg_actions = 0
    #             avg_time = 0

    #         summary[anomaly][method] = {
    #             "SR": round(success_rate, 1),
    #             "NA": round(avg_actions, 1),
    #             "TT": round(avg_time, 1)
    #         }

    # # Convert summary to DataFrame
    # df = pd.DataFrame.from_dict({(i,j): summary[i][j] 
    #                             for i in summary.keys() 
    #                             for j in summary[i].keys()}, orient='index')

    # # # Display the result in a table
    # #ace_tools.display_dataframe_to_user(name="Anomaly Experiment Summary", dataframe=df)

    # print(summary)


    # exit()

    #########################################

    for run in range(10000 if args.log else 20):  # Run 10 times if logging, otherwise once
        robotic_system.run_counter = run + 1  # Update run counter
        #robotic_system.clear_image_directories()  # Clear images at the start of each run
        
        try:
            robotic_system.log_counter = 0  # Reset state counter for each run
            robotic_system.grasp_and_move()

        except Exception as e:
            print(f"Error in grasp_and_move: {e}")

        print(f"Logged {robotic_system.log_counter} successful states. | Run numer = {robotic_system.run_counter}")
        print("I am here at the end no anomalies were found!")
        time.sleep(1)
        robotic_system.stop_simulation()
        time.sleep(2)  
        robotic_system.reset_simulation()
        time.sleep(1)

    time.sleep(2) 
    robotic_system.stop_simulation()
    cv2.destroyAllWindows()

def process_tasks(task_queue):
    while True:
        task = task_queue.get()
        task()
        task_queue.task_done()

if __name__ == "__main__":
    main()
