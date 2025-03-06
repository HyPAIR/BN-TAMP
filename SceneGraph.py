import networkx as nx
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from queue import Queue
import math
from threading import Lock
import zmq  # Import ZMQ



class SceneGraph:
    def __init__(self, task_queue):

        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.graph = nx.DiGraph()
        self.initial_edges = []  # To keep track of initial edges
        self.updated_edges = []  # To keep track of new edges
        self.positions = self.get_fixed_positions()
        self.task_queue = task_queue
        #self.robotic_system = robotic_system
        # Track the last two boxes being manipulated
        self.last_two_boxes = []

        self.current_on_relationships = []
        self.object_postition = []
        self.lock = Lock()  # Lock to protect canvas operations


        # Map user-friendly node names to actual object names in CoppeliaSim
        self.node_to_object = {
            'Gripper': '/UR10/tip',
            'Table 1': '/T1_Dummy',
            'Goal': '/Goal_Plate',
            'Box A': '/Block_A',
            'Box B': '/Block_B',
            'Box C': '/Block_C',
            'Occ_Obj': '/Occ_Obj'
        }
        self.handle_to_node = {}

        # Initialize node colors
        self.node_colors = {
            'Gripper': 'red',
            'Table 1': 'green',
            'Goal': 'green',
            'Box A': 'blue',
            'Box B': 'blue',
            'Box C': 'blue'
        }

        # Initialize the graph with nodes and edges
        self.init_graph()
    
    def get_relationships(self):
        relationships = []
        for u, v, data in self.graph.edges(data=True):
            relationships.append([u, v, data])
        return relationships

    def get_fixed_positions(self):
        return {
            'Gripper': (2, 1),
            'Table 1': (0, 1),
            'Goal': (1, 1),
            'Box A': (0, 0),
            'Box B': (1, 0),
            'Box C': (2, 0)
        }

    def init_graph(self):
        nodes = [
            'Gripper',
            'Table 1',
            'Goal',
            'Box A',
            'Box B',
            'Box C'
        ]

        edges = [
            # ('Box A', 'Table 1', 'on'),
            # ('Box B', 'Box A', 'on'),
            # ('Box C', 'Box B', 'on'),
        ]

        for node in nodes:
            self.graph.add_node(node)

        for edge in edges:
            self.graph.add_edge(*edge[:2], label=edge[2])

        self.initial_edges = list(self.graph.edges)  # Save initial edges

        # Initialize handle to node mapping
        self.initialize_handle_to_node_mapping()

        #self.update_positions()

    # def initialize_handle_to_node_mapping(self):
    #     print("Initializing handle-to-node mapping...")
    #     #self.robotic_system.reset_to_initial_state()
        
    #     for node, object_name in self.node_to_object.items():
    #         retries = 3  # Try up to 3 times
    #         while retries > 0:
    #             try:
    #                 handle = self.sim.getObject(object_name)
    #                 self.handle_to_node[handle] = node
    #                 position = self.sim.getObjectPosition(handle, -1)
    #                 self.positions[node] = position
    #                 break  # If successful, exit the retry loop
    #             except zmq.error.ZMQError as e:
    #                 print(f"ZMQ Error while resetting {node}: {e}")
    #                 retries -= 1
    #                 if retries == 0:
    #                     print(f"Failed to initialize {node} after 3 attempts.")
    #                 else:
    #                     time.sleep(1)  # Wait a bit before retrying
    #             except Exception as e:
    #                 print(f"Error while initializing handle for {node}: {e}")
    #                 #exit()
    #                 break  # Exit on any other exception

    # def update_positions(self):
    #    # self.robotic_system.reset_to_initial_state()
    #     try:
    #         for node, object_name in self.node_to_object.items():
    #             try:
    #                 handle = self.sim.getObject(object_name)
    #             except zmq.error.ZMQError as e:  # Corrected line
    #                 print(f"ZMQ Error while resetting 2: {e}")
    #             position = self.sim.getObjectPosition(handle, -1)
    #             self.positions[node] = position  # Update the real position
    #     except Exception as e:
    #         print(f"Error updating positions: {e}")
    #         #exit()

    # def initialize_handle_to_node_mapping(self):
    #     for node, object_name in self.node_to_object.items():
    #         try:
    #             handle = self.sim.getObject(object_name)
    #             self.handle_to_node[handle] = node
    #         except zmq.error.ZMQError as e:
    #             print(f"ZMQ Error while resetting {object_name}: {e}")
    #             time.sleep(1)  # Wait for a moment and retry
    #             # Retry logic
    #             handle = self.sim.getObject(object_name)
    #             self.handle_to_node[handle] = node

    def initialize_handle_to_node_mapping(self):
        self.handle_to_node = {}  # Reset the handle-to-node mapping
        
        print("Initializing handle-to-node mapping...")

        # Wait for simulation to stabilize before accessing objects
        time.sleep(2)  # Adjust if necessary
        
        for node, object_name in self.node_to_object.items():
            try:
                # Check if the object exists before trying to access it
                if self.sim.getObject(object_name):
                    handle = self.sim.getObject(object_name)
                    self.handle_to_node[handle] = node
                    print(f"Handle for {node} ({object_name}) successfully retrieved: {handle}")
                else:
                    print(f"Object {object_name} for node {node} does not exist in the simulation.")
                    
                # Optionally store positions for further use
                position = self.sim.getObjectPosition(handle, -1)
                self.positions[node] = position  # Store the real position of the object
                
            except Exception as e:
                print(f"ZMQ Error while resetting {object_name}: {e}")
                # Optionally, reinitialize specific objects if needed after an error
                if object_name == '/UR10/tip':
                    print(f"Attempting to reinitialize {object_name}...")
                    self.reinitialize_ur10_tip()

    def reinitialize_ur10_tip(self):
        try:
            self.ur10_tip_handle = self.sim.getObject('/UR10/tip')
            if self.ur10_tip_handle:
                print(f"/UR10/tip reinitialized with handle: {self.ur10_tip_handle}")
            else:
                print("Error: /UR10/tip handle could not be reinitialized.")
        except Exception as e:
            print(f"Error during /UR10/tip reinitialization: {e}")



    def update_positions(self):
        try:
            for node, object_name in self.node_to_object.items():
                handle = self.sim.getObject(object_name)
                position = self.sim.getObjectPosition(handle, -1)
                self.positions[node] = position  # Update the real position
        except Exception as e:
            print(f"Error updating positions: {e}")

    def capture_image(self, filename):
         with self.lock:
            self.update_positions()  # Update positions before capturing the image
            self.task_queue.put(lambda: self._capture_image(filename))

    def _capture_image(self, filename):
        
        with self.lock:  # Ensure thread safety when capturing images
            plt.figure(figsize=(10, 10))  # Increase figure size for better visibility

            # Draw the graph with fixed positions (x, y for drawing purposes)
            pos_2d = self.get_fixed_positions()
            nx.draw(
                self.graph,
                pos=pos_2d,
                with_labels=True,
                node_color=[self.node_colors.get(node, 'gray') for node in self.graph.nodes],
                node_size=3000,
                font_size=12,
                font_color='white',
                font_weight='bold'
            )

            # Add text labels with actual positions
            for node, pos in self.positions.items():
                fixed_pos = pos_2d[node]
                plt.text(
                    fixed_pos[0], fixed_pos[1] - 0.2,  # Adjust the vertical position
                    s=f'[x = {pos[0]:.2f}, y = {pos[1]:.2f}, z = {pos[2]:.2f}]',
                    bbox=dict(facecolor='white', alpha=0.5),
                    horizontalalignment='center'
                )

            # Draw edge labels
            edge_labels = nx.get_edge_attributes(self.graph, 'label')
            nx.draw_networkx_edge_labels(self.graph, pos=pos_2d, edge_labels=edge_labels, font_color='red')

            # Calculate plot limits with fixed padding
            padding_x = 1.7  # Adjust this value as needed
            padding_y = 1.2  # Adjust this value as needed

            x_vals = [pos[0] for pos in self.positions.values()]
            y_vals = [pos[1] for pos in self.positions.values()]
            x_min, x_max = min(x_vals) - padding_x, max(x_vals) + padding_x
            y_min, y_max = min(y_vals) - padding_y, max(y_vals) + padding_y

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # Detect and display anomalies only if there are new edges
            if self.updated_edges:
                anomalies = self.detect_anomalies()
                for anomaly in anomalies:
                    plt.text(0.5, 0.05, anomaly, horizontalalignment='center', verticalalignment='center', 
                            transform=plt.gcf().transFigure, fontsize=12, color='red', 
                            bbox=dict(facecolor='white', alpha=0.5))

            plt.savefig(filename)
            plt.close()

    def update_graph(self, add_edges=[], remove_edges=[]):

        for edge in remove_edges:
            if self.graph.has_edge(*edge[:2]):
                self.graph.remove_edge(*edge[:2])

        for edge in add_edges:
            self.graph.add_edge(*edge[:2], label=edge[2])
            self.updated_edges.append(edge)  # Add to updated edges list

            u, v, label = edge

    def detect_anomalies(self):
            
            name_mapping = {
                'Box A': 'Block A',
                'Box B': 'Block B',
                'Box C': 'Block C',
                'Occ_Box': 'Occ_Obj'
            }
           
            self.last_two_boxes = self.current_on_relationships
            self.object_post = self.object_postition['block_positions']
            self.goal_post = self.object_postition['goal_position']
            #print(f"What justhappend? = {self.last_two_boxes}")
            anomalies = []
            for boxes in self.last_two_boxes:
                #u, v, = boxes
                u = boxes[0]
                v = boxes[1]
                if 'Table' in u or 'Table' in v or 'Gripper' in v or 'Gripper' in u or 'Occ' in u or 'Occ' in v: continue
                if 'Goal' in u or 'Goal' in v:
                    for k,val in self.object_post.items():
                        if name_mapping[u] == k:
                            position_u = val

                    position_v = self.goal_post
                    overturn_threshold = 0.06  # Not needed, just kept for reference
                    misalignment_threshold = 0.1

                    misalignment_1 = abs(position_u[0] - position_v[0])
                    misalignment_2 = abs(position_u[1] - position_v[1])    

                    print(f" (1) u = {u} pos = {position_u} | v = {v} pos = {position_v}")

                    # Check for overturn
                    if abs(position_u[2] - position_v[2]) > overturn_threshold or misalignment_1 > misalignment_threshold or misalignment_2 > misalignment_threshold:
                        print("I am here 3", u,v)
                        anomalies.append(f'Missed Goal: {u} is not in {v}')
                        self.anomaly_detected = True  # Set the anomaly flag
                        #continue
    
                else:

                    for k,val in self.object_post.items():
                        if name_mapping[u] == k:
                            position_u = val
                        elif name_mapping[v] == k:
                            position_v = val
                
                    misalignment_threshold = 0.015
                

                    overturn_threshold = 0.08  # Not needed, just kept for reference

                    print(f"(2) u = {u} pos = {position_u} | v = {v} pos = {position_v}")

                    # Check for misalignment
                    misalignment = []
                    misalignment_1 = abs(position_u[0] - position_v[0])
                    misalignment_2 = abs(position_u[1] - position_v[1])                    

                    if misalignment_1 >= misalignment_threshold:
                        misalignment.append(True)
                    else:
                        misalignment.append(False)  

                    if misalignment_2 >= misalignment_threshold:
                        misalignment.append(True)
                    else:
                        misalignment.append(False)
                    # Check for overturn
                    #if position_v[2] + overturn_threshold > position_u[2]:
                    if abs(position_u[2]-position_v[2]) < overturn_threshold:   
                        print("I am here 3", u,v)
                        # if u == 'Box B' and position_u[2] > 0.60 and len(self.last_two_boxes) == 2:
                        #     print(position_u[2])
                        #     exit()
                        anomalies.append(f'Overturn detected: {u} is not above {v} : diff = {abs(position_v[2]-position_u[2])}')
                        self.anomaly_detected = True  # Set the anomaly flag
                        #continue

                    if any(misalignment):
                        #print("I am here 4", u,v)
                        anomalies.append(f'Misalignment detected between {u} and {v}: '
                                        f'X diff={abs(position_u[0] - position_v[0]):.2f}, '
                                        f'Y diff={abs(position_u[1] - position_v[1]):.2f}')
                        self.anomaly_detected = True  # Set the anomaly flag
                        #continue

            return anomalies

    def run(self):
        pass
        # while True:
        #     self.capture_image(f'images2/scene_{int(time.time())}.png')
        #     time.sleep(1)

    def reset(self):
        self.updated_edges = []  # Reset the updated edges
        self.last_two_boxes = []  # Reset the last two boxes being manipulated
        self.graph.clear()  # Clear the existing graph
        self.current_on_relationships = []
        self.object_postition = []
        # Reinitialize the graph with nodes and initial edges
        self.initial_edges = []  # To keep track of initial edges
        self.updated_edges = []  # To keep track of new edges
        self.init_graph()
        time.sleep(0.5)

        
