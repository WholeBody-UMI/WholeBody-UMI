"""Fixed array layouts used by the released model.

The 70-column input ``feat`` stores 69 motion channels plus world yaw at 69.
The 73-column model representation keeps the first 69 channels, then appends
heading-local root xy at 69:71 and relative yaw cos/sin at 71:73.

Shared motion channels:
    0:4   Root tilt: sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1.
    4     Root yaw increment from this frame to the next, in radians.
    5:7   Left/right foot contacts.
    7:10  Root displacement to the next frame, in metres. Input xy is in the
          world frame; model xy is rotated into the window heading.
    10    Root height, in metres.
    11:40 G1 joint angles, in radians: left leg, right leg, waist, left/right arm.
    40:69 Joint angle increments to the next frame, in radians.

The 50-column hand condition contains relative position/rot6d at 0:18, initial
heights at 18:20, initial heading-aligned rot6d at 20:32, and pose increments
at 32:50. The latter are per-frame differences, not metres per second.

The 36-column qpos contains world root xyz at 0:3, root wxyz at 3:7, and
29 joint angles at 7:36. Hand pose inputs use xyz followed by wxyz as well.
"""

REFERENCE_FEATURE_DIM = 70
BASE_MOTION_CHANNELS = slice(0, 69)
REFERENCE_YAW = 69

ROOT_ROLL_SIN = 0
ROOT_ROLL_COS_OFFSET = 1
ROOT_PITCH_SIN = 2
ROOT_PITCH_COS_OFFSET = 3
ROOT_YAW_DELTA = 4
FOOT_CONTACTS = slice(5, 7)
ROOT_DELTA_X = 7
ROOT_DELTA_Y = 8
ROOT_DELTA_XY = slice(7, 9)
ROOT_HEIGHT = 10
JOINT_ANGLES = slice(11, 40)
ROOT_X = 69
ROOT_Y = 70
ROOT_XY = slice(69, 71)
ROOT_YAW_COS = 71
ROOT_YAW_SIN = 72

CONDITION_VELOCITY = slice(32, 50)
QPOS_DIM = 36
QPOS_ROOT_XY = slice(0, 2)
QPOS_ROOT_QUATERNION = slice(3, 7)
