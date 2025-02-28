#conda activate rl_env

import pyAgrum as gum
import json
import os
import pandas as pd
import numpy as np
import pyAgrum.lib.image as gumimage
import re
import matplotlib.pyplot as plt
import json

# Function to generate quantile bins for discretization
def get_quantile_bins(values, num_bins):
    return np.quantile(values, np.linspace(0, 1, num_bins + 1))

def discretize(value, bins, labels):
    return labels[np.digitize(value, bins) - 1]

# Function to save the learned BN and structure
def save_bn(learned_bn, filename):
    gum.saveBN(learned_bn, filename)
    print(f"Bayesian Network saved to {filename}")

# Function to load a previously saved BN
def load_bn(filename):
    return gum.loadBN(filename)

# Function to save BN graph as image
def save_bn_image(learned_bn, state_name):
    output_folder = 'bn_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_name = os.path.join(output_folder, f'learned_bn_{state_name}.png')
    gumimage.export(learned_bn, file_name, size='100')
    print(f"Bayesian network for {state_name} saved as {file_name}")

#Function to generate the data from the logs to use to generate the BN
def generate_data(logs):
    # Initialize storage for variables grouped by state
    state_data = {}

    # Assume logs are stored in a folder called 'logs/'
    log_folder = logs
    # Function to extract state and run from filename
    def extract_state_run(filename):
        match = re.search(r'Log_State_(\d+)_Run_(\d+)', filename)
        if match:
            state = int(match.group(1))
            run = int(match.group(2))
            return state, run
        return None, None

    # Loop through all the logs and extract information
    for idx, log_file in enumerate(os.listdir(log_folder)):
        state, run = extract_state_run(log_file)


        # if state == 12 and run > 2017:
        #     continue

        if state is None or run is None:
            continue  # Skip files that don't match the naming convention
        if run > 2000:
            continue
        if state not in state_data:
            state_data[state] = {
                'box_a_x_values': [], 'box_a_y_values': [], 'box_a_z_values': [],
                'box_b_x_values': [], 'box_b_y_values': [], 'box_b_z_values': [],
                'box_c_x_values': [], 'box_c_y_values': [], 'box_c_z_values': [],
                'gripper_x_values': [], 'gripper_y_values': [], 'gripper_z_values': [],
                'goal_x_values': [], 'goal_y_values': [], 'goal_z_values': [],
                'anomaly_type_values': [], 'relationships_values': [], 'actions_values': []
            }

        with open(os.path.join(log_folder, log_file), 'r') as f:
            log_data = json.load(f)
            
            # Check if relationships exist; if empty, skip this run
            relationships = log_data.get('relationships', [])
            # if not relationships:  # Skip runs with empty relationships
            #     continue
            
            # Extracting positions of the boxes
            state_data[state]['box_a_x_values'].append(log_data['block_positions']['Block A'][0])
            state_data[state]['box_a_y_values'].append(log_data['block_positions']['Block A'][1])
            state_data[state]['box_a_z_values'].append(log_data['block_positions']['Block A'][2])

            state_data[state]['box_b_x_values'].append(log_data['block_positions']['Block B'][0])
            state_data[state]['box_b_y_values'].append(log_data['block_positions']['Block B'][1])
            state_data[state]['box_b_z_values'].append(log_data['block_positions']['Block B'][2])

            state_data[state]['box_c_x_values'].append(log_data['block_positions']['Block C'][0])
            state_data[state]['box_c_y_values'].append(log_data['block_positions']['Block C'][1])
            state_data[state]['box_c_z_values'].append(log_data['block_positions']['Block C'][2])

            state_data[state]['gripper_x_values'].append(log_data['gripper_position'][0])
            state_data[state]['gripper_y_values'].append(log_data['gripper_position'][1])
            state_data[state]['gripper_z_values'].append(log_data['gripper_position'][2])

            state_data[state]['goal_x_values'].append(log_data['goal_position'][0])
            state_data[state]['goal_y_values'].append(log_data['goal_position'][1])
            state_data[state]['goal_z_values'].append(log_data['goal_position'][2])

            # Extracting action data (Pick, Place, Push)
            state_data[state]['actions_values'].append(log_data['action'] if log_data['action'] else 'None')

            # Extract relationships and store them in the list
            state_data[state]['relationships_values'].append(relationships)

            # Store anomalies (single or multiple)
            anomaly_types = [anomaly for anomaly in log_data.get('anomalies', ['None'])]
            state_data[state]['anomaly_type_values'].append(anomaly_types)

    sorted_state_data = {k: state_data[k] for k in sorted(state_data)}
    state_data = sorted_state_data

    return state_data


def sanitize_variable_name(var_name):
    """
    Replace spaces and other invalid characters with underscores or appropriate characters.
    """
    # Replace spaces and non-alphanumeric characters with underscores
    return re.sub(r'\W+', '_', var_name)  # \W+ matches any character that is not a letter, digit, or underscore


def determine_failure_state(positions):
    # Check BoxC position
    if positions == 'low':
        return 'Box_C_X_too_low'
    elif positions == 'high':
        return 'Box_C_X_too_high'
    # Similarly for Y, Z, and BoxB if needed
    return 'Success'


# Function to create a BN with structure learning and saves the BN to .bif
def create_bn(state_data, size, save_image, structure_learning_type='greedy_hill_climbing'):
    allBins = []
    for state in state_data.keys():
        print(f"\nProcessing State {state}...")

        # Create a new Bayesian network for each state
        bn = gum.BayesNet(f'AnomalyHandling_State_{state}')

        if state_data[state]['actions_values'][0] == 'place':
            print(f"In this state {state} we are doing place")

            # Labels for discretization
            labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

            #if state == 12: continue
            bin_size = size

            # Precompute the quantile bins for each variable
            pre_box_a_x_bins = get_quantile_bins(state_data[state-1]['box_a_x_values'], bin_size)
            pre_box_a_y_bins = get_quantile_bins(state_data[state-1]['box_a_y_values'], bin_size)
            pre_box_a_z_bins = get_quantile_bins(state_data[state-1]['box_a_z_values'], bin_size)
            
            post_box_a_x_bins = get_quantile_bins(state_data[state]['box_a_x_values'], bin_size)
            post_box_a_y_bins = get_quantile_bins(state_data[state]['box_a_y_values'], bin_size)
            post_box_a_z_bins = get_quantile_bins(state_data[state]['box_a_z_values'], bin_size)
            ####
            pre_box_b_x_bins = get_quantile_bins(state_data[state-1]['box_b_x_values'], bin_size)
            pre_box_b_y_bins = get_quantile_bins(state_data[state-1]['box_b_y_values'], bin_size)
            pre_box_b_z_bins = get_quantile_bins(state_data[state-1]['box_b_z_values'], bin_size)


            post_box_b_x_bins = get_quantile_bins(state_data[state]['box_b_x_values'], bin_size)
            post_box_b_y_bins = get_quantile_bins(state_data[state]['box_b_y_values'], bin_size)
            post_box_b_z_bins = get_quantile_bins(state_data[state]['box_b_z_values'], bin_size)
            ####
            pre_box_b_x_bins = get_quantile_bins(state_data[state-1]['box_b_x_values'], bin_size)
            pre_box_b_y_bins = get_quantile_bins(state_data[state-1]['box_b_y_values'], bin_size)
            pre_box_b_z_bins = get_quantile_bins(state_data[state-1]['box_b_z_values'], bin_size)

            post_box_b_x_bins = get_quantile_bins(state_data[state]['box_b_x_values'], bin_size)
            post_box_b_y_bins = get_quantile_bins(state_data[state]['box_b_y_values'], bin_size)
            post_box_b_z_bins = get_quantile_bins(state_data[state]['box_b_z_values'], bin_size)
            ####
            pre_box_c_x_bins = get_quantile_bins(state_data[state-1]['box_c_x_values'], bin_size)
            pre_box_c_y_bins = get_quantile_bins(state_data[state-1]['box_c_y_values'], bin_size)
            pre_box_c_z_bins = get_quantile_bins(state_data[state-1]['box_c_z_values'], bin_size)

            post_box_c_x_bins = get_quantile_bins(state_data[state]['box_c_x_values'], bin_size)
            post_box_c_y_bins = get_quantile_bins(state_data[state]['box_c_y_values'], bin_size)
            post_box_c_z_bins = get_quantile_bins(state_data[state]['box_c_z_values'], bin_size)
            ####

            pre_gripper_x_bins = get_quantile_bins(state_data[state-1]['gripper_x_values'], bin_size)
            pre_gripper_y_bins = get_quantile_bins(state_data[state-1]['gripper_y_values'], bin_size)
            pre_gripper_z_bins = get_quantile_bins(state_data[state-1]['gripper_z_values'], bin_size)

            post_gripper_x_bins = get_quantile_bins(state_data[state]['gripper_x_values'], bin_size)
            post_gripper_y_bins = get_quantile_bins(state_data[state]['gripper_y_values'], bin_size)
            post_gripper_z_bins = get_quantile_bins(state_data[state]['gripper_z_values'], bin_size)
      
            #exit()

            bins = {'Pre_Box_A_X':pre_box_a_x_bins,
                    'Pre_Box_A_Y':pre_box_a_y_bins,
                    'Pre_Box_A_Z':pre_box_a_z_bins,
                    'Pre_Box_B_X':pre_box_b_x_bins,
                    'Pre_Box_B_Y':pre_box_b_y_bins,
                    'Pre_Box_B_Z':pre_box_b_z_bins,
                    'Pre_Box_C_X':pre_box_c_x_bins,
                    'Pre_Box_C_Y':pre_box_c_y_bins,
                    'Pre_Box_C_Z':pre_box_c_z_bins,
                    'Pre_Gripper_X':pre_gripper_x_bins,
                    'Pre_Gripper_Y':pre_gripper_y_bins,
                    'Pre_Gripper_Z':pre_gripper_z_bins,
                    'Post_Box_A_X':post_box_a_x_bins,
                    'Post_Box_A_Y':post_box_a_y_bins,
                    'Post_Box_A_Z':post_box_a_z_bins,
                    'Post_Box_B_X':post_box_b_x_bins,
                    'Post_Box_B_Y':post_box_b_y_bins,
                    'Post_Box_B_Z':post_box_b_z_bins,
                    'Post_Box_C_X':post_box_c_x_bins,
                    'Post_Box_C_Y':post_box_c_y_bins,
                    'Post_Box_C_Z':post_box_c_z_bins,
                    'Post_Gripper_X':post_gripper_x_bins,
                    'Post_Gripper_Y':post_gripper_y_bins,
                    'Post_Gripper_Z':post_gripper_z_bins,
            }

            # Specify the file path for saving the JSON file
            output_file = f'discretized_state_{state}_bins.json'

            # Convert each ndarray in bins to a list if it's not already a list
            for key, value in bins.items():
                if isinstance(value, np.ndarray):  # Check if the value is a numpy ndarray
                    bins[key] = value.tolist()     # Convert ndarray to a list
     
            # # Save the bins dictionary to a JSON file
            # with open(output_file, 'w') as json_file:
            #     json.dump(bins, json_file, indent=4)
            
            allBins.append(bins)
            
            # Discretize positions using precomputed bins
            pre_box_a_x_discrete = [discretize(x, pre_box_a_x_bins, labels) for x in state_data[state-1]['box_a_x_values']]
            pre_box_a_y_discrete = [discretize(y, pre_box_a_y_bins, labels) for y in state_data[state-1]['box_a_y_values']]
            pre_box_a_z_discrete = [discretize(z, pre_box_a_z_bins, labels) for z in state_data[state-1]['box_a_z_values']]

            post_box_a_x_discrete = [discretize(x, post_box_a_x_bins, labels) for x in state_data[state]['box_a_x_values']]
            post_box_a_y_discrete = [discretize(y, post_box_a_y_bins, labels) for y in state_data[state]['box_a_y_values']]
            post_box_a_z_discrete = [discretize(z, post_box_a_z_bins, labels) for z in state_data[state]['box_a_z_values']]
            ####
            pre_box_b_x_discrete = [discretize(x, pre_box_b_x_bins, labels) for x in state_data[state-1]['box_b_x_values']]
            pre_box_b_y_discrete = [discretize(y, pre_box_b_y_bins, labels) for y in state_data[state-1]['box_b_y_values']]
            pre_box_b_z_discrete = [discretize(z, pre_box_b_z_bins, labels) for z in state_data[state-1]['box_b_z_values']]

            post_box_b_x_discrete = [discretize(x, post_box_b_x_bins, labels) for x in state_data[state]['box_b_x_values']]
            post_box_b_y_discrete = [discretize(y, post_box_b_y_bins, labels) for y in state_data[state]['box_b_y_values']]
            post_box_b_z_discrete = [discretize(z, post_box_b_z_bins, labels) for z in state_data[state]['box_b_z_values']]
            ####
            pre_box_c_x_discrete = [discretize(x, pre_box_c_x_bins, labels) for x in state_data[state-1]['box_c_x_values']]
            pre_box_c_y_discrete = [discretize(y, pre_box_c_y_bins, labels) for y in state_data[state-1]['box_c_y_values']]
            pre_box_c_z_discrete = [discretize(z, pre_box_c_z_bins, labels) for z in state_data[state-1]['box_c_z_values']]

            post_box_c_x_discrete = [discretize(x, post_box_c_x_bins, labels) for x in state_data[state]['box_c_x_values']]
            post_box_c_y_discrete = [discretize(y, post_box_c_y_bins, labels) for y in state_data[state]['box_c_y_values']]
            post_box_c_z_discrete = [discretize(z, post_box_c_z_bins, labels) for z in state_data[state]['box_c_z_values']]
            ####
            pre_gripper_x_discrete = [discretize(x, pre_gripper_x_bins, labels) for x in state_data[state-1]['gripper_x_values']]
            pre_gripper_y_discrete = [discretize(y, pre_gripper_y_bins, labels) for y in state_data[state-1]['gripper_y_values']]
            pre_gripper_z_discrete = [discretize(z, pre_gripper_z_bins, labels) for z in state_data[state-1]['gripper_z_values']]

            post_gripper_x_discrete = [discretize(x, post_gripper_x_bins, labels) for x in state_data[state]['gripper_x_values']]
            post_gripper_y_discrete = [discretize(y, post_gripper_y_bins, labels) for y in state_data[state]['gripper_y_values']]
            post_gripper_z_discrete = [discretize(z, post_gripper_z_bins, labels) for z in state_data[state]['gripper_z_values']]

            # Add position nodes (Top Layer)
            for axis in ['X', 'Y', 'Z']:
                bn.add(gum.LabelizedVariable(f'Pre_BoxA_{axis}', f'Pre-action Position of Box A on {axis}', labels))
                bn.add(gum.LabelizedVariable(f'Pre_BoxB_{axis}', f'Pre-action Position of Box B on {axis}', labels))
                bn.add(gum.LabelizedVariable(f'Pre_BoxC_{axis}', f'Pre-action Position of Box C on {axis}', labels))
                bn.add(gum.LabelizedVariable(f'Pre_Gripper_{axis}', f'Pre-action Position of Gripper on {axis}', labels))

                bn.add(gum.LabelizedVariable(f'Post_BoxA_{axis}', f'Post-action Position of Box A on {axis}', labels))
                bn.add(gum.LabelizedVariable(f'Post_BoxB_{axis}', f'Post-action Position of Box B on {axis}', labels))
                bn.add(gum.LabelizedVariable(f'Post_BoxC_{axis}', f'Post-action Position of Box C on {axis}', labels))
                bn.add(gum.LabelizedVariable(f'Post_Gripper_{axis}', f'Post-action Position of Gripper on {axis}', labels))
            
            # Add relationship nodes (Third Layer)
            unique_relationships = []
            _index = 0
            for rel in state_data[state]['relationships_values']:
                for r in rel:
                    if r not in unique_relationships:
                        unique_relationships.append(r)
            relationship_data = {}
            # Loop through the unique relationships to analyze them
            for rel in unique_relationships:
                relevant_objects = [rel[0], rel[1]]  # e.g., ['Box_A', 'Goal']
                relationship_states = ['Success', 'Unknown']  # Default to "Success"
                
                # Skip relationships with 'Table 1' (if they are irrelevant)
                if 'Table 1' in rel:
                    continue
                
                # Generate potential failure states for the relevant object
                for obj in [relevant_objects[0]]:  # Check only the first object for misalignment
                    obj = obj.replace(" ", "_")
                    relationship_states += [f'{obj}_X_too_low', f'{obj}_X_too_high',
                                            f'{obj}_Y_too_low', f'{obj}_Y_too_high',
                                            f'{obj}_Z_too_low']

                # Create a node in the Bayesian network for this relationship
                rel_var_name = f"{rel[0]}_{rel[1]}_{rel[2]['label']}"
                rel_var_name = rel_var_name.replace(" ", "_")
                relationship_node = gum.LabelizedVariable(rel_var_name, f'Relationship: {rel[0]} {rel[2]["label"]} {rel[1]}', relationship_states)
                
                bn.add(relationship_node)
                misalignment_threshold = 0.015  # Example threshold for misalignment
                overturn_threshold = 0.035

                # Function to determine the failure state based on positions
                def determine_failure_state(object1_pos, object2_pos):
                    
                    #print(f"obj1 = {rel[0]}, object1_pos['z'] = {object1_pos['z']} | obj2 = {rel[1]}, object2_pos['z'] = {object2_pos['z']}")
                    if object1_pos['x'] < object2_pos['x'] - misalignment_threshold:
                        return f'{object1_pos["name"].replace(" ", "_")}_X_too_low'
                    elif object1_pos['x'] > object2_pos['x'] + misalignment_threshold:
                        return f'{object1_pos["name"].replace(" ", "_")}_X_too_high'
                    elif object1_pos['y'] < object2_pos['y'] - misalignment_threshold:
                        return f'{object1_pos["name"].replace(" ", "_")}_Y_too_low'
                    elif object1_pos['y'] > object2_pos['y'] + misalignment_threshold:
                        return f'{object1_pos["name"].replace(" ", "_")}_Y_too_high'
                    elif abs(object1_pos['z'] - object2_pos['z']) < overturn_threshold:
                        return f'{object1_pos["name"].replace(" ", "_")}_Z_too_low'
                    else:
                        return 'Success'
                
                # Add the relationship data (Success or failure state)
                relationship_data[sanitize_variable_name(rel_var_name)] = [
                    determine_failure_state(
                        {'name': rel[0], 
                        'x': state_data[state][f'{rel[0].replace(" ", "_").lower()}_x_values'][i],
                        'y': state_data[state][f'{rel[0].replace(" ", "_").lower()}_y_values'][i],
                        'z': state_data[state][f'{rel[0].replace(" ", "_").lower()}_z_values'][i]},
                        {'name': rel[1], 
                        'x': state_data[state][f'{rel[1].replace(" ", "_").lower()}_x_values'][i],
                        'y': state_data[state][f'{rel[1].replace(" ", "_").lower()}_y_values'][i],
                        'z': state_data[state][f'{rel[1].replace(" ", "_").lower()}_z_values'][i]}
                    )
                    for i, relationships in enumerate(state_data[state]['relationships_values'])
                 
                    ]
            
            print(f"Pre_BoxA_X size: {len(pre_box_a_x_discrete)}")
            print(f"Pre_BoxA_Y size: {len(pre_box_a_y_discrete)}")
            print(f"Pre_BoxA_Z size: {len(pre_box_a_z_discrete)}")
            
            print(f"Pre_BoxB_X size: {len(pre_box_b_x_discrete)}")
            print(f"Pre_BoxB_Y size: {len(pre_box_b_y_discrete)}")
            print(f"Pre_BoxB_Z size: {len(pre_box_b_z_discrete)}")
            
            print(f"Pre_BoxC_X size: {len(pre_box_c_x_discrete)}")
            print(f"Pre_BoxC_Y size: {len(pre_box_c_y_discrete)}")
            print(f"Pre_BoxC_Z size: {len(pre_box_c_z_discrete)}")
            
            print(f"Pre_Gripper_X size: {len(pre_gripper_x_discrete)}")
            print(f"Pre_Gripper_Y size: {len(pre_gripper_y_discrete)}")
            print(f"Pre_Gripper_Z size: {len(pre_gripper_z_discrete)}")
            
            print(f"Post_BoxA_X size: {len(post_box_a_x_discrete)}")
            print(f"Post_BoxA_Y size: {len(post_box_a_y_discrete)}")
            print(f"Post_BoxA_Z size: {len(post_box_a_z_discrete)}")
            
            print(f"Post_BoxB_X size: {len(post_box_b_x_discrete)}")
            print(f"Post_BoxB_Y size: {len(post_box_b_y_discrete)}")
            print(f"Post_BoxB_Z size: {len(post_box_b_z_discrete)}")
            
            print(f"Post_BoxC_X size: {len(post_box_c_x_discrete)}")
            print(f"Post_BoxC_Y size: {len(post_box_c_y_discrete)}")
            print(f"Post_BoxC_Z size: {len(post_box_c_z_discrete)}")
            
            print(f"Post_Gripper_X size: {len(post_gripper_x_discrete)}")
            print(f"Post_Gripper_Y size: {len(post_gripper_y_discrete)}")
            print(f"Post_Gripper_Z size: {len(post_gripper_z_discrete)}")
            
            print(f"Relationship data size: {len(relationship_data)}")

            ### Creating the DataFrame for structure learning ###
            df = pd.DataFrame({
                # 'Pre_BoxA_X': pre_box_a_x_discrete,
                # 'Pre_BoxA_Y': pre_box_a_y_discrete,
                # 'Pre_BoxA_Z': pre_box_a_z_discrete,
                # 'Pre_BoxB_X': pre_box_b_x_discrete,
                # 'Pre_BoxB_Y': pre_box_b_y_discrete,
                # 'Pre_BoxB_Z': pre_box_b_z_discrete,
                # 'Pre_BoxC_X': pre_box_c_x_discrete,
                # 'Pre_BoxC_Y': pre_box_c_y_discrete,
                # 'Pre_BoxC_Z': pre_box_c_z_discrete,
                # 'Pre_Gripper_X': pre_gripper_x_discrete,
                # 'Pre_Gripper_Y': pre_gripper_y_discrete,
                # 'Pre_Gripper_Z': pre_gripper_z_discrete,
                'Post_BoxA_X': post_box_a_x_discrete,
                'Post_BoxA_Y': post_box_a_y_discrete,
                'Post_BoxA_Z': post_box_a_z_discrete,
                'Post_BoxB_X': post_box_b_x_discrete,
                'Post_BoxB_Y': post_box_b_y_discrete,
                'Post_BoxB_Z': post_box_b_z_discrete,
                'Post_BoxC_X': post_box_c_x_discrete,
                'Post_BoxC_Y': post_box_c_y_discrete,
                'Post_BoxC_Z': post_box_c_z_discrete,
                'Post_Gripper_X': post_gripper_x_discrete,
                'Post_Gripper_Y': post_gripper_y_discrete,
                'Post_Gripper_Z': post_gripper_z_discrete,
                **relationship_data  # Include relationships
                #**action_data,  # Include actions
                #**anomaly_data  # Include anomalies
            })

            # Structure learning
            learner = gum.BNLearner(df)
            learner.useGreedyHillClimbing()
            
            def add_mandatory_arc(bn, from_var, to_var):
                if from_var in bn.names() and to_var in bn.names():
                    learner.addMandatoryArc(from_var, to_var)
                else:
                    print(f"Warning: One of the variables {from_var} or {to_var} does not exist in the network.")
            
            def add_forbidden_arc(bn, from_var, to_var):
                if from_var in bn.names() and to_var in bn.names():
                    learner.addForbiddenArc(from_var, to_var)
            
            # Define all the variables for pre and post conditions
            pre_position_nodes = [f'Pre_Box{box}_{axis}' for box in ['A', 'B', 'C'] for axis in ['X', 'Y', 'Z']] + \
                                [f'Pre_Gripper_{axis}' for axis in ['X', 'Y', 'Z']]
                                
            post_position_nodes = [f'Post_Box{box}_{axis}' for box in ['A', 'B', 'C'] for axis in ['X', 'Y', 'Z']] + \
                                [f'Post_Gripper_{axis}' for axis in ['X', 'Y', 'Z']]

            # Smart loop to map corresponding pre-variables to post-variables
            # for pre in pre_position_nodes:
            #     # Derive the corresponding post-condition variable by replacing 'Pre' with 'Post'
            #     post = pre.replace('Pre', 'Post')
            #     add_mandatory_arc(bn, pre, post)
        
                
            # # Learn the Bayesian network for a given state, then save it
            learned_bn = learner.learnBN()

            # Check for variables with insufficient modalities
            check_variable_modalities(learned_bn)

            # Remove variables with fewer than two modalities before saving/loading
            filter_single_state_variables(learned_bn)

            # test.append(learned_bn)
            # # Save the BN as a .bif file
            save_bn(learned_bn, f"bn_{state}.bif")

            # # Optionally, save the BN image
            if save_image:
                save_bn_image(learned_bn, f'{state}')
    return allBins
    
# Function for VE Inference
def variable_elimination(bn, evidence, target):
    ie = gum.LazyPropagation(bn)
    ie.setEvidence(evidence)
    result = ie.posterior(target)
    return result

# Function for MAP Inference
def map_inference(bn, evidence, targets):
    ie = gum.LazyPropagation(bn)
    ie.setEvidence(evidence)
    most_probable_values = {}
    
    for target in targets:
        posterior = gum.getPosterior(bn, evs=evidence, target=target)
        most_probable_value_tuple = posterior.argmax()
        most_probable_value = most_probable_value_tuple[0][0]
        most_probable_values[target] = most_probable_value
    
    return most_probable_values

# Function to handle setting evidence
def set_evidence(evidence_data):
    return evidence_data

# Function to check variable modalities and print warnings for any issues
def check_variable_modalities(bn):  
    for var in bn.names():
        variable = bn.variable(var)
        if variable.domainSize() < 2:
            print(f"Warning: Variable '{var}' has less than two states (Modalities: {variable.domainSize()}).")

def filter_single_state_variables(bn):
    # Loop through variables and remove any with fewer than two states
    variables_to_remove = [var for var in bn.names() if bn.variable(var).domainSize() < 2]
    for var in variables_to_remove:
        bn.erase(var)
    if variables_to_remove:
        print(f"Removed variables with only one state: {variables_to_remove}")

# Evaluate Other Anomalies
def evaluate_anomalies(learned_bn):
    # List of relationships/anomalies to evaluate
    anomalies = ['Box_B_Box_C_on', 'Box_B_Goal_on', 'Box_C_Goal_on']
    
    for relationship in anomalies:
        print(f"\nEvaluating {relationship}:\n")
        
        # Get and print the CPT (Conditional Probability Table) for the anomaly
        print(f"CPT of {relationship}:")
        print(learned_bn.cpt(relationship))
        
        # Perform Variable Elimination (VE) to get the marginal distribution
        print(f"\nPerforming Variable Elimination (VE) for {relationship}...")
        result_ve = variable_elimination(learned_bn, {}, relationship)
        print(f"Result of VE for {relationship}: \n{result_ve}")
        
        # Perform MAP Inference to find the most probable explanation
        print(f"\nPerforming MAP Inference for {relationship}...")
        targets = ['Pre_BoxB_X', 'Pre_BoxB_Y', 'Pre_BoxB_Z', 'Pre_BoxC_X', 'Pre_BoxC_Y', 'Pre_BoxC_Z', relationship]
        result_map = map_inference(learned_bn, {}, targets)
        print(f"MAP Inference result for {relationship}: {result_map}")

def process_new_state_data(new_state_data, bins, bin_size):
    bin = bins

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    labels = labels[:bin_size+1]
    result = {}

    for state in new_state_data.keys():
        print(f"Processing state {state} + {state+1}...")
        # Discretize the pre and post conditions for Box A, B, C, and Gripper
        pre_box_a_x_discrete = [discretize(x, bin['Pre_Box_A_X'], labels) for x in new_state_data[state]['box_a_x_values']]
        pre_box_a_y_discrete = [discretize(x, bin['Pre_Box_A_Y'], labels) for x in new_state_data[state]['box_a_y_values']]
        pre_box_a_z_discrete = [discretize(x, bin['Pre_Box_A_Z'], labels) for x in new_state_data[state]['box_a_z_values']]

        # Repeat for all relevant variables in the new_state_data
        pre_box_b_x_discrete = [discretize(x, bin['Pre_Box_B_X'], labels) for x in new_state_data[state]['box_b_x_values']]
        pre_box_b_y_discrete = [discretize(x, bin['Pre_Box_B_Y'], labels) for x in new_state_data[state]['box_b_y_values']]
        pre_box_b_z_discrete = [discretize(x, bin['Pre_Box_B_Z'], labels) for x in new_state_data[state]['box_b_z_values']]

        pre_box_c_x_discrete = [discretize(x, bin['Pre_Box_C_X'], labels) for x in new_state_data[state]['box_c_x_values']]
        pre_box_c_y_discrete = [discretize(x, bin['Pre_Box_C_Y'], labels) for x in new_state_data[state]['box_c_y_values']]
        pre_box_c_z_discrete = [discretize(x, bin['Pre_Box_C_Z'], labels) for x in new_state_data[state]['box_c_z_values']]

        pre_gripper_x_discrete = [discretize(x, bin['Pre_Gripper_X'], labels) for x in new_state_data[state]['gripper_x_values']]
        pre_gripper_y_discrete = [discretize(x, bin['Pre_Gripper_Y'], labels) for x in new_state_data[state]['gripper_y_values']]
        pre_gripper_z_discrete = [discretize(x, bin['Pre_Gripper_Z'], labels) for x in new_state_data[state]['gripper_z_values']]

        # Relationships and anomalies
        pre_relationships = new_state_data[state]['relationships_values']
        pre_anomalies = new_state_data[state]['anomaly_type_values']

        # Add Post states (for post conditions, similar to pre but for next state)
        post_state = state+1   # Assuming post state is state+1

        post_box_a_x_discrete = [discretize(x, bin['Post_Box_A_X'], labels) for x in new_state_data[post_state]['box_a_x_values']]
        post_box_a_y_discrete = [discretize(x, bin['Post_Box_A_Y'], labels) for x in new_state_data[post_state]['box_a_y_values']]
        post_box_a_z_discrete = [discretize(x, bin['Post_Box_A_Z'], labels) for x in new_state_data[post_state]['box_a_z_values']]
        
        post_box_b_x_discrete = [discretize(x, bin['Post_Box_B_X'], labels) for x in new_state_data[post_state]['box_b_x_values']]
        post_box_b_y_discrete = [discretize(x, bin['Post_Box_B_Y'], labels) for x in new_state_data[post_state]['box_b_y_values']]
        post_box_b_z_discrete = [discretize(x, bin['Post_Box_B_Z'], labels) for x in new_state_data[post_state]['box_b_z_values']]


        post_box_c_x_discrete = [discretize(x, bin['Post_Box_C_X'], labels) for x in new_state_data[post_state]['box_c_x_values']]
        post_box_c_y_discrete = [discretize(x, bin['Post_Box_C_Y'], labels) for x in new_state_data[post_state]['box_c_y_values']]
        post_box_c_z_discrete = [discretize(x, bin['Post_Box_C_Z'], labels) for x in new_state_data[post_state]['box_c_z_values']]
        
        post_gripper_x_discrete = [discretize(x, bin['Post_Gripper_X'], labels) for x in new_state_data[post_state]['gripper_x_values']]
        post_gripper_y_discrete = [discretize(x, bin['Post_Gripper_Y'], labels) for x in new_state_data[post_state]['gripper_y_values']]
        post_gripper_z_discrete = [discretize(x, bin['Post_Gripper_Z'], labels) for x in new_state_data[post_state]['gripper_z_values']]

        # Relationships and anomalies
        post_relationships = new_state_data[post_state]['relationships_values']
        post_anomalies = new_state_data[post_state]['anomaly_type_values']

        # Add all information to the result dictionary for comparison

        result[f"State_{state}"] = {
            # 'Pre_BoxA_X': pre_box_a_x_discrete[0],
            # 'Pre_BoxA_Y': pre_box_a_y_discrete[0],
            # 'Pre_BoxA_Z': pre_box_a_z_discrete[0],
            'Pre_BoxB_X': pre_box_b_x_discrete[0],
            'Pre_BoxB_Y': pre_box_b_y_discrete[0],
            'Pre_BoxB_Z': pre_box_b_z_discrete[0],
            'Pre_BoxC_X': pre_box_c_x_discrete[0],
            'Pre_BoxC_Y': pre_box_c_y_discrete[0],
            'Pre_BoxC_Z': pre_box_c_z_discrete[0],
            'Pre_Gripper_X': pre_gripper_x_discrete[0],
            'Pre_Gripper_Y': pre_gripper_y_discrete[0],
            'Pre_Gripper_Z': pre_gripper_z_discrete[0],
            'Relationships': pre_relationships,
            'Anomalies': pre_anomalies
        }

        result[f"State_{state+1}"] = {
            # 'Post_BoxA_X': post_box_a_x_discrete[0],
            # 'Post_BoxA_Y': post_box_a_y_discrete[0],
            # 'Post_BoxA_Z': post_box_a_z_discrete[0],
            'Post_BoxB_X': post_box_b_x_discrete[0],
            'Post_BoxB_Y': post_box_b_y_discrete[0],
            'Post_BoxB_Z': post_box_b_z_discrete[0],
            'Post_BoxC_X': post_box_c_x_discrete[0],
            'Post_BoxC_Y': post_box_c_y_discrete[0],
            'Post_BoxC_Z': post_box_c_z_discrete[0],
            'Post_Gripper_X': post_gripper_x_discrete[0],
            'Post_Gripper_Y': post_gripper_y_discrete[0],
            'Post_Gripper_Z': post_gripper_z_discrete[0],
            'Relationships': post_relationships,
            'Anomalies': post_anomalies
        }
        break

    return result

def RCA(pre_state, post_state, evidence, state, targets,learned_bn):

    """
    Perform Root Cause Analysis (RCA) to determine the likely cause of an anomaly based on the pre/post states.
    
    Parameters:
    - pre_state (dict): Discrete values representing the pre-action state (e.g., positions of objects and relationships)
    - post_state (dict): Discrete values representing the post-action state (e.g., positions of objects and relationships)
    - anomaly (str): The specific anomaly to be analyzed
    - learned_bn: The trained Bayesian network model to perform inference
    
    Returns:
    - root_cause (dict): A dictionary describing the likely root cause of the anomaly
    """

    print(f"\nPerforming Root Cause Analysis for {evidence}:{state}...\n")

    # Set anomaly as evidence
    evidence = {evidence: state}  # Evidence that the anomaly is present

    # Run MAP inference
    #post_variables = [key for key in post_state.keys() if key.startswith('Post_')]
    post_variables = targets
    post_result_map = map_inference(learned_bn, evidence, post_variables)  # Targeting post-state variables

    print("\nPost_State VE Values:")
    for k,v in post_state.items():
        if k in post_variables:
            print(f"Var = {v}, VE = {variable_elimination(learned_bn, evidence, k)}")

    print("\nPost_State Discerte Values:")
    for k,v in post_state.items():
        print(f"{k}: {v}")

    print("\nMAP Inference Post Variables Results:")
    for var, value in post_result_map.items():
        print(f"{var}: {value}")

    print("\n#####################################\n")

    # Compare MAP results with post-state and flag differences
    print("\nComparing MAP results with post-state variables...\n")
    root_causes = []
    for var in post_variables:
        post_value = post_state[var]
        map_value = post_result_map.get(var)

        # Compare post-state with MAP result
        if str(map_value[var]) == post_value:
            root_causes.append((var, post_value, map_value[var]))
            print(f"Match found: {var} = {post_value} (MAP: {map_value[var]})")

    if root_causes:
        print("\nIdentified Root Causes:")
        for cause in root_causes:
            print(f"Variable: {cause[0]}, Value: {cause[1]}, MAP Predicted: {cause[2]}")
    else:
        print("No significant root cause identified from changed variables.")
    
def MAP(pre_state, post_state, evidence, state, targets,learned_bn):
    print(f"\nPerforming Recovery for {evidence}:{state}...\n")

    '''
    Need to make this one more cleaver later, where I can just focus on ensuring that I split the objects
    then return the changes for each object where if a certain variable dont need to change keep it as same as the intial state...
    '''
    
    # Step 1: Set the anomaly to 'False' to find a recovery configuration
    evidence = {evidence: state}
    
    # Step 2: Perform MAP inference targeting all post-state variables
    #targets = [key for key in post_state.keys() if key.startswith("Post_")]

    result_map = map_inference(learned_bn, evidence, targets)
    
    # Step 3: Print out the post-state and MAP results
    print("\nPost_State Values (Current):")
    for key, value in post_state.items():
        if key.startswith("Post_"):
            print(f"{key}: {value}")
    
    print("\nMAP Inference Results (Recovery Target):")
    for target in targets:
        print(f"{target}: {result_map[target]}")
    
    # Step 4: Compare MAP results with the current post-state to identify changes
    print("\nComparing MAP results with post-state variables for recovery...\n")
    
    corrective_actions = {}
    goal = []
    for target in targets:
        post_value = post_state[target]
        map_value = result_map[target][target]
        
        if str(post_value) != str(map_value):
            corrective_actions[target] = {"Current Value": post_value, "Suggested Recovery": map_value}
            goal.append([target, map_value])
            print(f"Correction Needed: {target} - Current Value: {post_value}, Suggested Value: {map_value}")
    
    # Step 5: Return the corrective actions (or print them)
    if corrective_actions:
        print("\nIdentified Corrective Actions:")
        for key, values in corrective_actions.items():
            print(f"Variable: {key}, Current: {values['Current Value']}, Suggested: {values['Suggested Recovery']}")
    else:
        print("No corrective actions needed - the current state is already optimal.")
    
    return corrective_actions,goal

def transistionChanges():
    #Generate this function later
    #We want to see if Pre->Post changes
    #This can be useful for anomalies that happens over time
    #Perhaps even if we collapse the stacking or object we dont place changes etc. 
    pass

# Main workflow
def main():
    logs = 'logs/logs1'
    state_data = generate_data(logs)
    # Generate the Bayesian Network, Structure Learning, Save bif and images for all states
    #bins = create_bn(state_data, 10, True,structure_learning_type='greedy_hill_climbing')

    with open('discretized_state_8_bins.json', 'r') as json_file:
        loaded_bins = json.load(json_file)

    # print(len(bins[0]))
    # print(len(loaded_bins))
    # exit()
    #exit()

    # Load the learned BN (if saved)
   
    newLog = 'logs/test_log'
  
    new_state_data = generate_data(newLog)

    new_data = process_new_state_data(new_state_data,loaded_bins,10)
    
    # evaluate_anomalies(learned_bn)

    learned_bn = gum.loadBN('bn_8.bif')

    # evidence = {'Box_B_Box_C_on': 'Success'}  # Evidence that the anomaly is present
    # result_ve = variable_elimination(learned_bn, evidence, 'Post_BoxB_Z')
    # print(f"Result of VE: {result_ve}")
    
    evidence = {'Box_B_Box_C_on': 'Success'}
    result_ve = variable_elimination(learned_bn, evidence, 'Post_BoxB_X')
    print(f"Result of VE with Success: {result_ve}")

    # Recovery(pre_state, post_state, evidence, state, targets,learned_bn)
    # targets = [key for key in post_state.keys() if key.startswith("Post_")]
    
    targets = ['Post_BoxB_X','Post_BoxB_Y','Post_BoxB_Z']
    RCA(new_data['State_7'], new_data['State_8'], 'Box_B_Box_C_on', 'Box_B_X_too_low',targets ,learned_bn)
    print("\n\n Perform MAP")
    _,map = MAP(new_data['State_7'], new_data['State_8'], 'Box_B_Box_C_on', 'Success', targets, learned_bn)
    

    goal_state = []
    for box,val in map:
        #print(box,val)
        goal = loaded_bins['Post_Box_'+str(box).split('Box')[1]][val]
        goal_state.append(goal)
        
    print("\n\n")
    print(goal_state)

    
    #exit()
    
    # evidence = {'Box_B_Goal_on': 'Box_B_X_too_high'}
    # result_ve = variable_elimination(learned_bn, evidence, 'Post_BoxB_X')
    # print(f"Result of VE with Box_B_X_Too_High: {result_ve}")

    # evidence = {'Box_B_Goal_on': 'Box_B_X_too_low'} 
    # result_ve = variable_elimination(learned_bn, evidence, 'Post_BoxB_X')
    # print(f"Result of VE with Box_B_X_Too_Low: {result_ve}")

    # evidence = {'Box_B_Goal_on': 'Success'}
    # targets = ['Post_BoxB_X', 'Post_BoxB_Y']
    # result_map = map_inference(learned_bn, evidence, targets)
    # print("MAP Inference result with evidience: Success :", result_map)

    '''
    Something is indeed incorrect, it seems
    that when there is missalignement the MAP will somehow say that it perfers the Z to be as low
    as possible which I do not think is the case, but sure it can be the case if we know overturn -> always misalignement! 
    Perhaps we need to sepetate the anomalies to include mixtures? even specific stuff ugh. 
    '''

    # Perform MAP Inference
    # evidence = {'Anomaly_Misalignment': False}  # Evidence that the anomaly is present
    # targets = ['Pre_BoxB_Z', 'Post_BoxB_Z']
    # result_map = map_inference(learned_bn, evidence, targets)
    # print("MAP Inference result:", result_map)


    # Create a new BN and copy the variables
    #new_bn = gum.BayesNet("Modified_BN")



if __name__ == "__main__":
    main()