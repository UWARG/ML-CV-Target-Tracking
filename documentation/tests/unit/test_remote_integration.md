# Target Tracking: Integration Testing Protocol

This document outlines the two-phase testing protocol for the OAK-D vision-to-MAVLink tracking pipeline.

## Phase 1: SITL Verification (Software-Only)

**Goal:** Validate the Proportional Velocity Controller ($K_p$) and MAVLink packet construction without requiring physical camera hardware.

**Prerequisites:**
- Docker installed on the host machine.
- `pymavlink` installed in the local Python environment.

**Execution Steps:**
1. Boot the virtual ArduPilot flight controller using Docker:
   ```bash
   docker run --name ardupilot-sitl --rm -it -p 5762:5760 radarku/ardupilot-sitl --out tcpin:0.0.0.0:5760
   ```
2. Run the mock integration script:
   ```bash
   python3 test_remote_integration.py
   ```

**Success Criteria:** 
The terminal streams successful `SET_POSITION_TARGET_LOCAL_NED` commands. The virtual drone should calculate forward velocities (max 2.0m/s) when the target is distant, hover (0.0m/s) at exactly 2.0 meters, and command negative velocities to back up if the target breaches the 2.0m safety radius.