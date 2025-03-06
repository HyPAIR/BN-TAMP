import time
import json
import os
import random

class ExperimentRunner:
    def __init__(self, robotic_system, num_trials=3, log_dir="/home/yazz/Desktop/BN-TAMP/_data/"):
        self.robotic_system = robotic_system
        self.num_trials = num_trials
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "experiment_results_test.json")
        
        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Clear the log file at the start of each run
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

   # Define a function to handle re-running in case of IK failure
    def run_with_ik_retries(self, method_function, anomaly_type, max_retries=5):
        for i in range(max_retries):
            print(f"Running {method_function.__name__} for {anomaly_type} (Attempt {i+1}/{max_retries})")
            
            # Reset the timer before each retry
            start_time = time.time()
            
            success, action_count = method_function(anomaly_type)
            
            # If success is NOT -1, we either solved it or failed due to max attempts (not IK)
            if success != -1:
                end_time = time.time()
                elapsed_time = end_time - start_time  # Compute elapsed time for successful attempt
                return success, action_count, round(elapsed_time, 2)  # Return with correct time

            print(f"IK failure detected in {method_function.__name__}. Retrying... Attempt {i+1}/{max_retries}")
            self.setup_anomaly(anomaly_type)  # Reset simulation
            time.sleep(1)

        # If all retries fail, return a valid failure response instead of None
        print(f"All {max_retries} IK retries failed for {method_function.__name__}.")
        return False, 0, float('inf')  # Indicate failure with a high time penalty


    def run_experiment(self, anomaly_type, method):
        """
        Runs a series of experiments for a given anomaly type using either:
        - "nominal" (randomized brute-force recovery)
        - "BN_PDDLStream" (Bayesian Network + PDDLStream recovery)
        """
        results = []
        
        for trial in range(self.num_trials):
            trial_id = trial + 1
            print(f"Running trial {trial_id} for {anomaly_type} using {method}...")
            
            # Initialize anomaly condition
            self.setup_anomaly(anomaly_type)
            time.sleep(1)
            
            # Start tracking time
            start_time = time.time()
            action_count = 0
            success = 0
            
            # Run the experiment with retries for IK failures
            if method == "nominal_1":
                success, action_count, elapsed_time = self.run_with_ik_retries(self.run_nominal_recovery_1, anomaly_type)
            elif method == "nominal_2":
                success, action_count, elapsed_time = self.run_with_ik_retries(self.run_nominal_recovery_2, anomaly_type)
            elif method == "BN_PDDLStream":
                success, action_count, elapsed_time = self.run_with_ik_retries(self.run_pddlstream_recovery, anomaly_type)
            else:
                raise ValueError("Invalid method specified")

            # Compute elapsed time
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            
            # Measure stack stability after recovery
            #stability = self.check_stability()
            
            # Log results
            trial_result = {
                "trial_id": trial_id,
                "anomaly_type": anomaly_type,
                "method": method,
                "success": int(success),
                "time_taken": round(elapsed_time, 2),
                "actions_used": action_count
                #"stack_stability": stability
            }
            results.append(trial_result)
            
            # Save intermediate logs
            self.save_results(trial_result)
            
            # Reset the simulation
            self.robotic_system.reset_simulation()
            time.sleep(1)
        
        print(f"Completed {self.num_trials} trials for {anomaly_type} using {method}.")
        return results
    

    def anomaly_checker(self):
        self.robotic_system.update_scene_graph()
        self.robotic_system.scene_graph.current_on_relationships = self.robotic_system.scene_graph.get_relationships()
        time.sleep(0.5)
        self.robotic_system.scene_graph.object_postition = self.robotic_system.get_current_state()
        anomalies = self.robotic_system.scene_graph.detect_anomalies()
        print(self.robotic_system.scene_graph.get_relationships())
        print(f"Is there really an anomaly? {anomalies}")

        if anomalies == []:
            return False
        else:
            return True


    def setup_anomaly(self, anomaly_type):

        self.robotic_system.stop_simulation()
        time.sleep(2)  
        self.robotic_system.reset_simulation()
        time.sleep(1)
        self.robotic_system.turn_off_box_dynamics()
        time.sleep(2)
     

        """Sets the simulation to the predefined anomaly state."""
        if anomaly_type == "M_B":
            self.robotic_system.set_initial_positions(
                block_a_pos=[-0.7, 0.4, 0.17],
                block_b_pos=[-0.73, 0.4, 0.30],
            )
        elif anomaly_type == "O_B":
            self.robotic_system.set_initial_positions(
                block_a_pos=[-0.7, 0.4, 0.17],
                block_b_pos=[-0.75, 0.4, 0.30]
            )
        elif anomaly_type == "M_A_B_M_B_C":
            self.robotic_system.set_initial_positions(
                block_a_pos=[-0.70, 0.4, 0.20],
                block_b_pos=[-0.72, 0.4, 0.33],
                block_c_pos=[-0.7, 0.4, 0.45]
            )
        elif anomaly_type == "M_B_O_C":
            self.robotic_system.set_initial_positions(
                block_a_pos=[-0.7, 0.4, 0.20],
                block_b_pos=[-0.73, 0.4, 0.33],
                block_c_pos=[-0.64, 0.4, 0.45]
            )
        elif anomaly_type == "O_B_O_C":
            self.robotic_system.set_initial_positions(
                block_a_pos=[-0.7, 0.4, 0.17],
                block_b_pos=[-0.75, 0.4, 0.30],
                block_c_pos=[-0.68, 0.4, 0.42]
            )
        time.sleep(3)
        self.robotic_system.turn_on_box_dynamics()
        time.sleep(2)
        self.robotic_system.turn_off_box_dynamics()
        time.sleep(1)

    def fix_misalignment(self, object_handle, target_handle, action_count, max_attempts=10):
        for attempt in range(max_attempts):
            move_checker = self.robotic_system.new_pick(object_handle)
            if move_checker == False:
                print("I failed to pick and failed")
                return -1, action_count
            time.sleep(0.5)
            move_checker = self.robotic_system.new_place(object_handle, target_handle)
            if move_checker == False:
                print("I failed to place and failed")
                return -1, action_count
            time.sleep(0.5)
            action_count += 2
            
            if not self.anomaly_checker():
                return 1, action_count  # If successful, return immediately

            if attempt == max_attempts - 1:
                return 0, action_count  # If max attempts reached, return failure

        
    def run_nominal_recovery_1(self, anomaly_type):
        """Runs the Nominal (Random) Recovery approach."""
        action_count = 0
        success = 0
        print("I am about to try recover using nominal planner")        
        return success, action_count

    def run_nominal_recovery_2(self, anomaly_type):
        """Runs the Nominal (Random) Recovery approach."""
        action_count = 0
        success = 0
        print("I am trying to recovery using BN+PDDLStream")
        time.sleep(0.5)
        return success, action_count
    
    def run_pddlstream_recovery(self, anomaly_type):
        """Runs the BN + PDDLStream Recovery approach."""

        action_count = 0
        success = 0
        print("I am trying to recovery using BN+PDDLStream")

        anomalies = self.anomaly_checker()
        print(f"what is anomalies = {anomalies}")
   
        current_state =   self.robotic_system.get_current_state()
        #Save the current state
        if anomaly_type == "M_B" or anomaly_type == "O_B":
            self.robotic_system.log_current_state("B")
        else:
            self.robotic_system.log_current_state("B,C")
        
        self.robotic_system.run_planner()

        initial_state_file = '/home/yazz/Desktop/BN-TAMP/_data/my_plan.json'
        
        with open(initial_state_file, 'r') as f:
            plan_data = json.load(f)
        
        print(plan_data)

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
                if int(object) == 3:
                    block = self.robotic_system.block_a_handle
                if int(object) == 4:
                    block = self.robotic_system.block_b_handle
                if int(object) == 5:
                    block = self.robotic_system.block_c_handle

                # Debug: Print the pose and grasp
                print(f"Pick action: Object={details['object']}, Pose={pose}, Grasp={grasp}")
                move_checker = self.robotic_system.new_pick(block)
                if move_checker == False:
                    print("I am inside PDDLStream and failed to pick")
                    time.sleep(1)
                    return -1, action_count
                action_count+=1

            #Process 'place' action
            if action == "place":

                '''
                Figure out why it does not move to place location? it should indeed
                be different to the pick placement?
                '''

                pose = details["pose"][0]  # The [x, y, z] part of the pose
                grasp = details["grasp"][0]  # The [x, y, z] part of the grasp

                # Debug: Print the pose and grasp
                print(f"Place action: Object={details['object']}, Pose={pose}, Grasp={grasp}")

                # Example usage of your move_to_position function in CoppeliaSim
                # Move to the object's pose
                print(f"Moving to pose...{pose}")
                move_checker = self.robotic_system.move_to_position([-pose[0], pose[1], pose[2]+0.1])  # Replace 'self' with your class instance or context
                if move_checker == False:
                    print("I am inside PDDLStream failed to place")
                    time.sleep(1)
                    return -1, action_count
                print("I have now moved to the location to place")
                time.sleep(2)
                self.robotic_system.place(block)
                print("I have placed the block")
                time.sleep(2)
                action_count+=1

        time.sleep(0.5)
        return 1, action_count

    def check_stability(self):
        """Checks stack stability after recovery."""
        time.sleep(5)  # Wait to observe if stack remains stable
        return 100 if not self.robotic_system.scene_graph.detect_anomalies() else 50
    
    def save_results(self, trial_result):
        """Saves the results to a JSON file."""
        file_path = self.log_file
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(trial_result)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

# Example usage:
# experiment_runner = ExperimentRunner(robotic_system)
# experiment_runner.run_experiment("M_B", "nominal")
