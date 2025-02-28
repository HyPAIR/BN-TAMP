#!/bin/bash

#./launch_tmux.sh
#Launch the above, change the port to match the port in main then run either main1-5 which will run headless directly on the given scene!

#./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23000



# Set the starting port and the number of instances
start_port=23001  # Starting port
instances=1       # Number of instances (adjust this as needed)
wait_time=1       # Time to wait between launching each instance (adjustable)

# Loop through the desired number of instances
for ((i=0; i<$instances; i++))
do
    # Calculate the port number for this instance
    port=$((start_port + i))

    # Open a new terminal tab and execute the CoppeliaSim command
    gnome-terminal --tab -- bash -c "
        cd CoppeliaSim_Edu_V4_7_0_rev4_Ubuntu22_04/;
        echo 'Launching CoppeliaSim on port $port';
        ./coppeliaSim.sh -H /home/yazz/Desktop/Active_Simulate_Plan_Before_New_Scene/CoppeliaSim_To_PDDLStream_Backup.ttt -GzmqRemoteApi.rpcPort=$port;
        exec bash"
    
    # Wait for the specified time before launching the next instance
    sleep $wait_time
done
