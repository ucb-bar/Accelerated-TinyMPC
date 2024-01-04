"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).
# read_action()
"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import re

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

import time
import subprocess


NOISE = False

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
# DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
#DEFAULT_SIMULATION_FREQ_HZ = 240
#DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_SIMULATION_FREQ_HZ = 400
DEFAULT_CONTROL_FREQ_HZ = 50
#DEFAULT_DURATION_SEC = 12
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run_tinympc_system(observation):
    # os.system("spike pk ./examples/example_quadrotor_hovering", formated_obs)
    #base_command = "spike pk ./build/examples/example_quadrotor_hovering"
    base_command = "./build/examples/example_quadrotor_hovering_single"
    # base_command = "./build/examples/example_quadrotor_tracking_pybullet"
    formated_obs =extract_state_inputs(observation)
    full_command = f"{base_command} {formated_obs}"
    result = subprocess.run(full_command, shell=True, text=True, capture_output=True)
    print(f"RESULT {type(result)} {result}")
    return result.stdout

    # Convert observation to string and send. 
    # TinyMPC: x,y,z Rodriguez parameters
    # RiskyBIrd(ESP): PID: quaternion, angular rate

def quaternion_to_RodriguesParam(q):
    """
    Convert a quaternion into RodriguesParam (r1,r2,r3)
    q: Quaternion represented as [q1, q2, q3, q4]
    """
    # Extract the values from the quaternion
    q1, q2, q3, q4 = q
    # RodriguesParam calculation
    r1 = q1/q4
    r2 = q2/q4
    r3 = q3/q4
    return r1,r2,r3

def extract_state_inputs(observation):
    """
    quaternion, angular rate -> x,y,z Rodriguez parameters
    Extracts the state from the Gym observation.
    observation: ndarray from Gym environment
    """
    print(f'TYPE OF OBS: {type(observation)}')
    observation = observation[0]
    # Position and linear velocities
    x, y, z, vx, vy, vz = observation[0], observation[1], observation[2], observation[10], observation[11], observation[12]

    # Quaternion to RodriguesParam angles
    quaternion = observation[3:7]
    r1,r2,r3 = quaternion_to_RodriguesParam(quaternion)

    # Angular velocities
    dphi, dtheta, dpsi = observation[13], observation[14], observation[15]

    formatted_state_string = f"{x} {y} {z} {r1} {r2} {r3} {vx} {vy} {vz} {dphi} {dtheta} {dpsi}"
    # print (f"Formatted_State_String =========== {type(formatted_state_string)} {formatted_state_string}")
    return formatted_state_string
def calculate_rpm(env, command_output_thrusts):
    """Read the action returned from the TinyMPC"""
    normalized_thrusts_string = get_action_from_command(command_output_thrusts)

    normalized_thrusts = parse_string_to_numbers(normalized_thrusts_string)
    print(f"======= Normalized thrusts {normalized_thrusts}" )
    
    # normalized_forces is the normalized motor thrust 0-1 with 1 ~ 65535 pwm ~ 60 grams ~ 0.59 N, First, muliply by  0.59N and then add the 0.0662N hover force for each prop when simulating our drone.
    max_thrust_N = 0.58/4 # Max thrust in Newtons corresponding to normalized value 1
    actual_thrusts = (normalized_thrusts + 0.583) * max_thrust_N

    # Clip negative thrust values to 0
    actual_thrusts_clipped = np.clip(actual_thrusts, 0, None)    
    print(f"======= Calculated force {actual_thrusts_clipped}" )
    print(f"======== KF {env.KF}, MAX_THRUST_ENV {env.MAX_THRUST}") 
    rpm = np.sqrt(actual_thrusts_clipped / env.KF)

    rpm =  np.array(rpm)
    
    return rpm

def get_action_from_command(output):

    # Split the output into lines and get the last line
    lines = output.strip().split('\n')
    last_line = lines[-1]

    # Extract the numbers from the last line
    numbers_str = last_line.split()[-4:]  # Split the last line and take the last 4 elements
    numbers = [float(num) for num in numbers_str]

    return np.array(numbers)

# def calculate_rpm(forces, env):
#     """
#     Calculate the RPM for given forces.
    
#     forces: A numpy array of forces applied.
#     KF: The constant factor relating forces to the square of RPM.
#     """
#     # Ensure forces are non-negative
#     forces = np.maximum(forces, 0)
    
#     # Calculate RPM
#     rpm = np.sqrt(forces / env.KF)
#     return rpm

def parse_string_to_numbers(input_data):
    """
    Parse an input (string or numpy array) to extract four numbers and return them in a NumPy array.
    The input string is expected to contain four numbers separated by spaces or colons.
    """
    # Handle the case where the input is a NumPy array
    if isinstance(input_data, np.ndarray):
        # Flatten the array and take the first four elements
        float_numbers = list(input_data.flatten()[:4])
    elif isinstance(input_data, str):
        # Replace colons with spaces and split the string
        parts = input_data.replace(":", " ").split()

        # Filter and convert to float
        float_numbers = [float(part) for part in parts if part.replace('.', '', 1).replace('-', '', 1).isdigit()]
    else:
        raise TypeError("Input must be a string or a numpy array")

    # Ensure we have exactly 4 numbers, pad with zeros if necessary/f
    while len(float_numbers) < 4:
        float_numbers.append(0.0)

    return np.array(float_numbers[:4])



def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    # H = .1
    H = .02
    H_STEP = .05
    R = .3

    MOTOR_TAU = 0.7 * 10e-6

    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0.1, 0.1,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()


    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        #### Step the simulation ###################################
        obs, _, _, _, _ = env.step(action)
        print(f"obs: {obs}")
        forces_string = run_tinympc_system(obs)
        print(f"=========== forces {forces_string}")
        rpm = calculate_rpm(env, forces_string)
        print(f"=========== RPM {rpm}")
        action[0] = rpm
        

        #### Go to the next way point and loop #####################
        # for j in range(num_drones):
        #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        # #### Log the simulation ####################################
        logger.log(drone=0,
                    timestamp=i/env.CTRL_FREQ,
                    state=obs[0],
                    control=np.hstack([TARGET_POS[wp_counters[0], 0:2], INIT_XYZS[0, 2], INIT_RPYS[0, :], np.zeros(6)])
                    # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                    )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    # #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("TinyMPC") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
        
        
        
        # print(f'motor observation: {obs[0][16:20]}')
        # action[j,:] = update_motor_rpm(obs[0][16:20], action[j,:], MOTOR_TAU,control_freq_hz)
        # print(f'esp32  action (filtered)[{j}]: {action[j,:]}')

