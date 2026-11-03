import numpy as np
import cv2 as cv

# Used Generative AI for comments and docstrings. As well as corner selection instructions.

# ============================================================
# SETTINGS
# ============================================================

video_file = "bolt_video.MOV"

# Measured physical center-to-corner distance
R_hex = 0.035  # meters

# Downscale video for easier display / faster processing.
# Change to None if you want original resolution.
PROCESS_WIDTH = 960

# ============================================================
# SHIFT FUNCTION
# ============================================================

shift_y = -420

def shift_frame(frame):

    M = np.float32([
        [1, 0, 0],
        [0, 1, shift_y]
    ])

    shifted = cv.warpAffine(
        frame,
        M,
        (frame.shape[1], frame.shape[0])
    )

    return shifted

# ============================================================
# RESIZE FUNCTION
# ============================================================

def resize_frame(frame):

    if PROCESS_WIDTH is None:
        return frame

    h, w = frame.shape[:2]

    scale = PROCESS_WIDTH / float(w)

    new_h = int(h * scale)

    return cv.resize(
        frame,
        (PROCESS_WIDTH, new_h)
    )


# ============================================================
# EULER ANGLE FUNCTION
# ============================================================

def rotation_matrix_to_euler(R):

    sy = np.sqrt(
        R[0, 0] ** 2 +
        R[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        roll = np.arctan2(
            R[2, 1],
            R[2, 2]
        )

        pitch = np.arctan2(
            -R[2, 0],
            sy
        )

        yaw = np.arctan2(
            R[1, 0],
            R[0, 0]
        )

    else:

        roll = np.arctan2(
            -R[1, 2],
            R[1, 1]
        )

        pitch = np.arctan2(
            -R[2, 0],
            sy
        )

        yaw = 0.0

    return np.degrees([
        roll,
        pitch,
        yaw
    ])


# ============================================================
# KNOWN PHYSICAL HEXAGON
# ============================================================
#
# Coordinate frame:
#
# center = (0,0,0)
#
# Points correspond to:
#
# start at RIGHTMOST corner
# then move CLOCKWISE around the cap.
#

sqrt3 = np.sqrt(3.0)

object_points = np.array([

    # P0 upper-left
    [-R_hex / 2.0, -sqrt3 * R_hex / 2.0, 0.0],

    # P1 upper-right
    [ R_hex / 2.0, -sqrt3 * R_hex / 2.0, 0.0],

    # P2 rightmost
    [ R_hex, 0.0, 0.0],

    # P3 lower-right
    [ R_hex / 2.0,  sqrt3 * R_hex / 2.0, 0.0],

    # P4 lower-left
    [-R_hex / 2.0,  sqrt3 * R_hex / 2.0, 0.0],

    # P5 leftmost
    [-R_hex, 0.0, 0.0]

], dtype=np.float32)


# Physical center of cap
center_object = np.array(
    [[0.0, 0.0, 0.0]],
    dtype=np.float32
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv.VideoCapture(video_file)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open {video_file}"
    )


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:
    raise RuntimeError(
        "Could not read first frame."
    )

first_frame = resize_frame(first_frame)

# Apply vertical shift
first_frame = shift_frame(first_frame)

height, width = first_frame.shape[:2]


# ============================================================
# APPROXIMATE CAMERA INTRINSICS
# ============================================================
#
# These are ONLY for the qualitative test.
#
# Do not trust absolute XYZ or angles yet.
#

fx = float(width)
fy = float(width)

cx = width / 2.0
cy = height / 2.0

camera_matrix = np.array([
    [fx,  0.0, cx],
    [0.0, fy,  cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

dist_coeffs = np.zeros(
    (5, 1),
    dtype=np.float64
)


# ============================================================
# MANUAL CORNER SELECTION
# ============================================================

selected_points = []


def mouse_callback(event, x, y, flags, param):

    if (
        event == cv.EVENT_LBUTTONDOWN
        and len(selected_points) < 6
    ):

        selected_points.append(
            [float(x), float(y)]
        )

        print(
            f"Selected P{len(selected_points)-1}: "
            f"({x}, {y})"
        )


cv.namedWindow(
    "Select 6 Hex Corners",
    cv.WINDOW_NORMAL
)

cv.setMouseCallback(
    "Select 6 Hex Corners",
    mouse_callback
)


print()
print("Select six TOP-FACE corners.")
print()
print("Start at the RIGHTMOST corner.")
print("Then continue CLOCKWISE.")
print()
print("Press R to reset.")
print("Press Q to quit.")
print()


while len(selected_points) < 6:

    display = first_frame.copy()

    for i, point in enumerate(selected_points):

        px = int(point[0])
        py = int(point[1])

        cv.circle(
            display,
            (px, py),
            6,
            (0, 0, 255),
            -1
        )

        cv.putText(
            display,
            f"P{i}",
            (px + 8, py - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv.imshow(
        "Select 6 Hex Corners",
        display
    )

    key = cv.waitKey(20) & 0xFF

    if key == ord("r"):
        selected_points.clear()

    elif key == ord("q"):
        cap.release()
        cv.destroyAllWindows()
        raise SystemExit


cv.destroyWindow(
    "Select 6 Hex Corners"
)


# ============================================================
# INITIALIZE OPTICAL FLOW
# ============================================================

previous_gray = cv.cvtColor(
    first_frame,
    cv.COLOR_BGR2GRAY
)

previous_points = np.array(
    selected_points,
    dtype=np.float32
).reshape(-1, 1, 2)


# Lucas-Kanade parameters
lk_params = dict(

    winSize=(31, 31),

    maxLevel=4,

    criteria=(
        cv.TERM_CRITERIA_EPS |
        cv.TERM_CRITERIA_COUNT,
        50,
        0.001
    )
)


# Previous PnP pose
previous_rvec = None
previous_tvec = None


# ============================================================
# VIDEO TRACKING LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = resize_frame(frame)

    # Apply SAME vertical shift
    frame = shift_frame(frame)

    gray = cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # TRACK SIX CORNERS
    # --------------------------------------------------------

    current_points, status, error = (
        cv.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            previous_points,
            None,
            **lk_params
        )
    )


    if current_points is None:

        print("Optical flow failed.")
        break


    status = status.reshape(-1)

    good_count = np.sum(status)


    if good_count != 6:

        print(
            f"Tracking lost: "
            f"only {good_count}/6 corners remain."
        )

        for i in range(6):

            if status[i] == 0:
                print(f"Lost corner: P{i}")

        break


    image_points = (
        current_points
        .reshape(6, 2)
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # PNP
    # --------------------------------------------------------

    if previous_rvec is None:

        success, rvec, tvec = cv.solvePnP(

            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,

            flags=cv.SOLVEPNP_IPPE
        )

    else:

        # Use previous pose as starting guess.
        # This helps temporal continuity.

        rvec_guess = previous_rvec.copy()
        tvec_guess = previous_tvec.copy()

        success, rvec, tvec = cv.solvePnP(

            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,

            rvec_guess,
            tvec_guess,

            True,

            flags=cv.SOLVEPNP_ITERATIVE
        )


    if success:

        previous_rvec = rvec.copy()
        previous_tvec = tvec.copy()


        # ----------------------------------------------------
        # PROJECT FIXED PHYSICAL CENTER
        # ----------------------------------------------------

        projected_center, _ = cv.projectPoints(

            center_object,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )

        center_pixel = (
            projected_center
            .reshape(2)
        )

        center_pixel_int = (
            int(round(center_pixel[0])),
            int(round(center_pixel[1]))
        )


        # ----------------------------------------------------
        # DRAW TRACKED HEX
        # ----------------------------------------------------

        polygon = image_points.astype(
            np.int32
        ).reshape((-1, 1, 2))

        cv.polylines(
            frame,
            [polygon],
            True,
            (0, 255, 0),
            2
        )


        # Draw each tracked vertex
        for i, pt in enumerate(image_points):

            px = int(round(pt[0]))
            py = int(round(pt[1]))

            cv.circle(
                frame,
                (px, py),
                5,
                (255, 0, 0),
                -1
            )

            cv.putText(
                frame,
                f"P{i}",
                (px + 5, py - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


        # ----------------------------------------------------
        # DRAW PNP CENTER
        # ----------------------------------------------------

        cv.circle(
            frame,
            center_pixel_int,
            8,
            (0, 0, 255),
            -1
        )


        # ----------------------------------------------------
        # DRAW COORDINATE FRAME
        # ----------------------------------------------------

        cv.drawFrameAxes(
            frame,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            0.025
        )


        # ----------------------------------------------------
        # 6-DOF DATA
        # ----------------------------------------------------

        X = float(tvec[0, 0])
        Y = float(tvec[1, 0])
        Z = float(tvec[2, 0])

        R_matrix, _ = cv.Rodrigues(
            rvec
        )

        roll, pitch, yaw = (
            rotation_matrix_to_euler(
                R_matrix
            )
        )


        cv.putText(
            frame,
            (
                f"XYZ: "
                f"{X:.3f}, "
                f"{Y:.3f}, "
                f"{Z:.3f} m"
            ),
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv.putText(
            frame,
            (
                f"RPY: "
                f"{roll:.1f}, "
                f"{pitch:.1f}, "
                f"{yaw:.1f} deg"
            ),
            (20, 55),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv.putText(
            frame,
            "RED = fixed physical center",
            (20, 80),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv.imshow(
        "Bolt Pose Tracking",
        frame
    )

    key = cv.waitKey(20) & 0xFF

    if key == ord("q"):
        break


    # --------------------------------------------------------
    # UPDATE TRACKER
    # --------------------------------------------------------

    previous_gray = gray.copy()

    previous_points = (
        current_points.copy()
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv.destroyAllWindows()