import time
from pymavlink import mavutil


def calculate_velocity_commands(target_x_m, target_y_m, target_z_m):
    desired_z = 2.0
    desired_x = 0.0
    kp_z = 0.5
    kp_x = 0.5

    error_z = target_z_m - desired_z
    error_x = target_x_m - desired_x

    vx = max(-2.0, min(2.0, kp_z * error_z))
    vy = max(-2.0, min(2.0, kp_x * error_x))

    return vx, vy, 0.0, 0.0


def send_velocity_command(master, vx, vy, vz, yaw_rate):
    type_mask = 0b010111000111
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0,
        0,
        0,
        vx,
        vy,
        vz,
        0,
        0,
        0,
        0,
        yaw_rate,
    )


def main():
    print("Connecting to local SITL drone...")
    master = mavutil.mavlink_connection("tcp:127.0.0.1:5762")
    master.wait_heartbeat()
    print("Connected to virtual drone!")

    # A simulated sequence of camera detections over time
    # Format: (time_seconds, x_meters, y_meters, z_meters, description)
    simulated_target_path = [
        (2.0, 0.0, 0.0, 5.0, "Target appears 5m dead ahead"),
        (2.0, 0.0, 0.0, 4.0, "Target walks closer (4m)"),
        (2.0, 0.0, 0.0, 2.0, "Target hits safe distance (2m)"),
        (2.0, 1.0, 0.0, 2.0, "Target side-steps to the right (1m)"),
        (2.0, 0.0, 0.0, 1.0, "Target aggressively approaches (1m) - Drone should back up"),
    ]

    print("\nStarting Vision Simulation...\n")

    for duration, sim_x, sim_y, sim_z, description in simulated_target_path:
        print(f"\n--- {description} ---")
        end_time = time.time() + duration

        # Send commands continuously for 'duration' seconds, simulating a camera frame rate
        while time.time() < end_time:
            vx, vy, vz, yaw = calculate_velocity_commands(sim_x, sim_y, sim_z)
            send_velocity_command(master, vx, vy, vz, yaw)

            print(
                f"[Simulated Camera] Target Z: {sim_z:.1f}m, X: {sim_x:.1f}m -> Sending MAVLink Fwd: {vx:.2f}m/s, Right: {vy:.2f}m/s"
            )

            time.sleep(0.1)  # Simulate 10 FPS camera


if __name__ == "__main__":
    main()
