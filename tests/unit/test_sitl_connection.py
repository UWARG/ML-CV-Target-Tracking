from pymavlink import mavutil
import time

def test_connection():
    print("Attempting to connect to SITL via TCP...")
    # This matches the 'address' in your config.yaml
    master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')

    print("Waiting for MAVLink heartbeat (timeout in 10s)...")
    
    # Wait for the first heartbeat to confirm two-way communication
    msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=10.0)
    
    if msg is None:
        print("FAILED: No heartbeat received. The Docker container is blocking the port.")
    else:
        print(f"SUCCESS: Heartbeat received! System ID: {master.target_system}, Component ID: {master.target_component}")
        print(f"Vehicle Mode: {msg.custom_mode}")

if __name__ == "__main__":
    test_connection()