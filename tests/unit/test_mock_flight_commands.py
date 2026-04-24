def calculate_velocity_commands(target_x, target_y, target_z):
    desired_z = 2.0  # Safe distance: 2 meters away
    desired_x = 0.0  # Centered horizontally
    
    kp_z = 0.5  # Forward/back aggressiveness
    kp_x = 0.5  # Left/right aggressiveness
    
    error_z = target_z - desired_z
    error_x = target_x - desired_x
    
    vx = kp_z * error_z   # Positive = fly forward
    vy = kp_x * error_x   # Positive = fly right
    vz = 0.0      
    yaw_rate = 0.0 
    
    # Safety caps
    vx = max(-2.0, min(2.0, vx))
    vy = max(-2.0, min(2.0, vy))
    
    return vx, vy, vz, yaw_rate

def run_mock_tests():
    print("--- RUNNING MOCK MAVLINK VELOCITY TESTS ---\n")
    
    scenarios = [
        {"name": "Target is far away (5m ahead, centered)", "x": 0.0, "z": 5.0},
        {"name": "Target is too close (1m ahead, centered)", "x": 0.0, "z": 1.0},
        {"name": "Target is at perfect safe distance (2m ahead)", "x": 0.0, "z": 2.0},
        {"name": "Target is 2m away, but drifted to the right (1m right)", "x": 1.0, "z": 2.0},
        {"name": "Extreme outlier (False positive 50m away)", "x": 0.0, "z": 50.0},
    ]
    
    for s in scenarios:
        vx, vy, vz, yaw = calculate_velocity_commands(s["x"], 0.0, s["z"])
        print(f"SCENARIO: {s['name']}")
        print(f"  Input:  Target at X={s['x']}m, Z={s['z']}m")
        print(f"  Output: Command Drone to fly -> Forward: {vx:.2f} m/s | Right: {vy:.2f} m/s\n")

if __name__ == "__main__":
    run_mock_tests()