"""
Drone command interface using MAVLink.

Translates tracked-object positions (camera-relative meters) into
MAVLink commands that move the drone toward a target or yaw to face it.

Coordinate frames
-----------------
Camera (OAK-D):  x = right,   y = down,    z = forward
Drone body:      x = forward, y = right,   z = down
NED (world):     x = north,   y = east,    z = down

The camera is assumed to be forward-facing and aligned with the body
frame, so the conversion is:
    body_x =  camera_z   (forward)
    body_y =  camera_x   (right)
    body_z =  camera_y   (down)

Body-to-NED requires the drone's current yaw heading.

MAVLink messages used
---------------------
- SET_POSITION_TARGET_LOCAL_NED (mavutil #84)
    Move to an offset in the local NED frame.
- COMMAND_LONG  MAV_CMD_CONDITION_YAW (#115)
    Rotate to face a heading.
"""

import math
import logging
from typing import Optional

from pymavlink import mavutil

logger = logging.getLogger(__name__)


class DroneCommander:
    """
    Sends MAVLink commands to the flight controller.

    Usage::

        commander = DroneCommander("tcp:localhost:5762")
        commander.connect()
        commander.move_to_target(forward=2.0, right=0.5, down=0.3)
        commander.face_target(forward=2.0, right=0.5)
    """

    def __init__(
        self,
        connection_string: str,
        baud_rate: int = 57600,
    ) -> None:
        self._connection_string = connection_string
        self._baud_rate = baud_rate
        self._conn: Optional[mavutil.mavlink_connection] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish MAVLink connection and wait for a heartbeat.
        """
        logger.info(
            "Connecting to flight controller at %s …",
            self._connection_string,
        )
        self._conn = mavutil.mavlink_connection(
            self._connection_string,
            baud=self._baud_rate,
        )
        self._conn.wait_heartbeat()
        logger.info(
            "Connected  (system %d, component %d).",
            self._conn.target_system,
            self._conn.target_component,
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def camera_to_body(
        cam_x: float,
        cam_y: float,
        cam_z: float,
    ) -> tuple:
        """
        Convert camera-frame coords to drone body-frame coords.

        Camera:  x=right, y=down, z=forward
        Body:    x=forward, y=right, z=down

        Returns:
            (body_x, body_y, body_z) in metres.
        """
        return cam_z, cam_x, cam_y

    @staticmethod
    def body_to_ned(
        body_x: float,
        body_y: float,
        body_z: float,
        yaw_rad: float,
    ) -> tuple:
        """
        Rotate body-frame offset into NED using the drone's current yaw.

        Args:
            body_x: Forward (m).
            body_y: Right (m).
            body_z: Down (m).
            yaw_rad: Current heading in radians (0 = north, CW positive).

        Returns:
            (north, east, down) in metres.
        """
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        north = body_x * cos_yaw - body_y * sin_yaw
        east = body_x * sin_yaw + body_y * cos_yaw
        down = body_z
        return north, east, down

    # ------------------------------------------------------------------
    # Movement commands
    # ------------------------------------------------------------------

    def move_to_target(
        self,
        forward: float,
        right: float,
        down: float,
        yaw_rad: float = 0.0,
    ) -> None:
        """
        Send a position-offset command in the local NED frame.

        The offsets are body-frame distances to the target; they are
        rotated into NED before sending.

        Args:
            forward: Body-frame forward offset (m).
            right: Body-frame right offset (m).
            down: Body-frame down offset (m).
            yaw_rad: Drone's current yaw (rad) for NED conversion.
        """
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        north, east, d = self.body_to_ned(forward, right, down, yaw_rad)

        # type_mask: ignore velocity, acceleration, yaw, yaw_rate
        # bits 0-2 = position (use), 3-5 = velocity (ignore),
        # 6-8 = acceleration (ignore), 10 = yaw (ignore),
        # 11 = yaw_rate (ignore)
        type_mask = 0b0000_1111_1111_1000  # 0x0FF8 — only position used

        self._conn.mav.set_position_target_local_ned_send(
            0,                                          # time_boot_ms
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # offset from current pos
            type_mask,
            north,                                      # x (north)
            east,                                       # y (east)
            d,                                          # z (down)
            0, 0, 0,                                    # vx, vy, vz (ignored)
            0, 0, 0,                                    # afx, afy, afz (ignored)
            0, 0,                                       # yaw, yaw_rate (ignored)
        )

        logger.debug(
            "move_to_target  body=(%.2f, %.2f, %.2f)  NED=(%.2f, %.2f, %.2f)",
            forward, right, down, north, east, d,
        )

    def face_target(
        self,
        forward: float,
        right: float,
    ) -> None:
        """
        Yaw the drone to face a target given its body-frame offset.

        Computes the required heading change from the body-frame
        forward/right offset and sends MAV_CMD_CONDITION_YAW.

        Args:
            forward: Body-frame forward offset (m) to target.
            right: Body-frame right offset (m) to target.
        """
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # angle from drone nose to target (positive = clockwise)
        angle_deg = math.degrees(math.atan2(right, forward))

        # direction: 1 = CW, -1 = CCW
        direction = 1 if angle_deg >= 0 else -1
        angle_deg = abs(angle_deg)

        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,                  # confirmation
            angle_deg,          # param1: target angle (degrees)
            0,                  # param2: angular speed (deg/s, 0 = default)
            direction,          # param3: 1=CW, -1=CCW
            1,                  # param4: 1=relative, 0=absolute
            0, 0, 0,           # params 5-7 (unused)
        )

        logger.debug(
            "face_target  body_offset=(%.2f, %.2f)  yaw_change=%.1f° %s",
            forward, right, angle_deg, "CW" if direction == 1 else "CCW",
        )

    # ------------------------------------------------------------------
    # High-level helper
    # ------------------------------------------------------------------

    def track_target(
        self,
        cam_x: float,
        cam_y: float,
        cam_z: float,
        yaw_rad: float = 0.0,
        move: bool = True,
        face: bool = True,
    ) -> None:
        """
        Convenience method: convert camera coords and send both
        move + face commands for a single tracked object.

        Args:
            cam_x: Camera-frame x (right, m).
            cam_y: Camera-frame y (down, m).
            cam_z: Camera-frame z (forward/depth, m).
            yaw_rad: Drone's current heading (rad).
            move: Whether to send a position command.
            face: Whether to send a yaw command.
        """
        forward, right, down = self.camera_to_body(cam_x, cam_y, cam_z)

        if face:
            self.face_target(forward, right)

        if move:
            self.move_to_target(forward, right, down, yaw_rad)
