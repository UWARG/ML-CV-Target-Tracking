from pymavlink import mavutil
import time

def send_velocity_command(master, vx, vy, vz, yaw_rate):
    """
    Sends a velocity command to the flight controller.
    vx: Forward/Back velocity (m/s)
    vy: Right/Left velocity (m/s)
    vz: Down/Up velocity (m/s) (Note: Negative is UP in NED coordinate frame)
    yaw_rate: Turn rate (rad/s)
    """
    # Bitmask to ignore position and acceleration, only use velocity and yaw rate
    type_mask = 0b010111000111  
    
    master.mav.set_position_target_local_ned_send(
        0,       # time_boot_ms (not used)
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED, # Frame relative to the drone's current heading
        type_mask,
        0, 0, 0, # x, y, z positions (ignored by mask)
        vx, vy, vz, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (ignored)
        0, yaw_rate # yaw, yaw_rate
    )

def calculate_velocity_commands(target_x, target_y, target_z):
    # Constants
    desired_z = 2.0  # Safe distance in meters
    desired_x = 0.0  # Centered horizontally
    
    kp_z = 0.5  # Forward/back aggressiveness
    kp_x = 0.5  # Left/right aggressiveness
    
    # Calculate errors
    error_z = target_z - desired_z   # Forward/backward error
    error_x = target_x - desired_x   # Lateral error
    
    # P‑controller outputs
    vx = kp_z * error_z   # Forward velocity (positive = fly forward)
    vy = kp_x * error_x   # Lateral velocity (positive = fly right)
    
    # We will ignore altitude (vz) and rotation (yaw) for this basic test
    vz = 0.0  
    yaw_rate = 0.0 
    
    # Safety cap: Don't let the drone fly faster than 2.0 m/s in any direction
    vx = max(-2.0, min(2.0, vx))
    vy = max(-2.0, min(2.0, vy))
    
    return vx, vy, vz, yaw_rate

def main():
    print("Connecting to SITL...")
    master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')
    master.wait_heartbeat()
    print("Connected.")

    # Simulate a target sitting 5 meters dead ahead, slightly to the right.
    simulated_target_z = 5.0  
    simulated_target_x = 1.0  
    simulated_target_y = 0.0  
    
    vx, vy, vz, yaw = calculate_velocity_commands(simulated_target_x, simulated_target_y, simulated_target_z)
    
    print(f"Calculated Velocities -> Forward: {vx}m/s, Right: {vy}m/s")
    
    # In a real loop, we would send this continuously.
    # send_velocity_command(master, vx, vy, vz, yaw)

if __name__ == "__main__":
    main()