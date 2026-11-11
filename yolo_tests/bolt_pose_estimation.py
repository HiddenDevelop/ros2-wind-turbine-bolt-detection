#!/usr/bin/env python3
"""
bolt_pose_estimation.py

Point-cloud stage for hex bolt localization: estimates each bolt's ORIENTATION
and its true CENTER (not just the visible surface), from YOLO instance masks
and Azure Kinect registered depth.

Front end (same as yolo_seg_depth_localization.py, duplicated so this node
runs on its own): full-resolution masks -> erosion -> undistorted per-pixel
back-projection -> MAD depth outlier rejection -> per-bolt 3D point cloud.

Point-cloud analysis per bolt:
  1. PCA on the points. The ratio of the two largest standard deviations
     ("elongation") says how the bolt is seen:
       - compact (elongation < threshold) -> HEAD-ON: head facing the camera
       - elongated                        -> SIDE-ON: bolt lying on its side
  2. HEAD-ON: a RANSAC plane fit finds the flat top of the head. The plane
     normal (flipped to face the camera) gives the bolt axis; the center is the
     head-top center moved half a head height into the bolt.
  3. SIDE-ON: the first principal component is the bolt axis. The wider end
     (larger cross-section) is taken as the head, so the axis points head -> tip.
     The center is the surface center moved one shank radius away from the
     camera, onto the bolt's axis.
  4. Detections that fit neither case cleanly are reported as UNKNOWN with the
     visible-surface center and no orientation.

Pose convention: the pose's +Z axis is the bolt axis, pointing head -> tip.
Rotation about the axis is not estimated (hex flat orientation is arbitrary).

Output frame: RGB camera optical frame (x right, y down, z forward), meters.

Topics
  pub  /hex_bolt/poses         geometry_msgs/PoseArray         (one pose per bolt)
  pub  /hex_bolt/pose_markers  visualization_msgs/MarkerArray  (axis arrow + label with ID/kind)
  pub  /hex_bolt/points        sensor_msgs/PointCloud2         (x, y, z, bolt_id) for RViz

ros2 run azure_kinect_ros2_driver azure_kinect_node --ros-args -p camera_info_topic:=/rgb/camera_info -p rgb_topic:=/rgb/image_raw -p depth_topic:=/depth_to_rgb/image_raw

"""

import math
import time
import traceback

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


# ---------------------------------------------------------------------------
# Front-end helpers (same as yolo_seg_depth_localization.py)
# ---------------------------------------------------------------------------

def build_ray_table(K, D, width, height):
    """Undistorted normalized ray (X/Z, Y/Z) for every pixel."""
    u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))
    pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
    # Iterate to convergence; the 5-iteration default leaves ~0.2 px error near
    # the edges with the Azure Kinect's rational distortion model.
    if hasattr(cv2, 'undistortPointsIter'):        # OpenCV 4.x
        rays = cv2.undistortPointsIter(pts, K, D, None, None, criteria)
    else:                                          # OpenCV 5.x: criteria is a keyword
        try:
            rays = cv2.undistortPoints(pts, K, D, criteria=criteria)
        except (TypeError, cv2.error):
            rays = cv2.undistortPoints(pts, K, D)
    rays = rays.reshape(height, width, 2)
    return rays[..., 0].copy(), rays[..., 1].copy()


def depth_to_meters(depth_image, encoding):
    if encoding in ('16UC1', 'mono16'):
        return depth_image.astype(np.float32) * 0.001
    if encoding == '32FC1':
        d = depth_image.astype(np.float32, copy=False)
        return np.where(np.isfinite(d), d, 0.0).astype(np.float32)
    raise ValueError(f'Unsupported depth encoding "{encoding}" (expected 16UC1 or 32FC1)')


def prepare_mask(mask_prob, image_shape, kernel):
    raw = (mask_prob > 0.5).astype(np.uint8)
    if raw.shape != image_shape:
        raw = cv2.resize(raw, (image_shape[1], image_shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    eroded = cv2.erode(raw, kernel) if kernel is not None else raw
    return raw.astype(bool), eroded.astype(bool)


def robust_surface_points(mask, depth_m, ray_x, ray_y,
                          mad_k=3.0, min_mad_m=0.003, min_pixels=30):
    sel = mask & (depth_m > 0.0)
    if int(sel.sum()) < min_pixels:
        return None
    z = depth_m[sel]
    med = float(np.median(z))
    sigma = max(1.4826 * float(np.median(np.abs(z - med))), min_mad_m)
    keep = np.abs(z - med) <= mad_k * sigma
    if int(keep.sum()) < min_pixels:
        return None
    z = z[keep]
    return np.column_stack([ray_x[sel][keep] * z, ray_y[sel][keep] * z, z])


# ---------------------------------------------------------------------------
# Point-cloud geometry (no ROS dependencies, so it can be unit-tested directly)
# ---------------------------------------------------------------------------

def principal_components(points):
    """Return (mean, eigenvalues descending, eigenvectors as columns)."""
    mean = points.mean(axis=0)
    cov = np.cov((points - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return mean, np.clip(evals[order], 0.0, None), evecs[:, order]


def fit_plane_ransac(points, inlier_thresh, iterations, rng, refine_iters=5):
    """RANSAC plane fit refined by SVD on the inliers.
    Returns (unit normal, point on plane, refined inlier mask, inlier RMS,
    support mask within inlier_thresh) or None."""
    n = len(points)
    if n < 3:
        return None
    best_inliers, best_count = None, 0
    for _ in range(iterations):
        p0, p1, p2 = points[rng.choice(n, 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        inliers = np.abs((points - p0) @ normal) < inlier_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_inliers, best_count = inliers, count
    if best_inliers is None or best_count < 3:
        return None
    # Least-squares refinement. The RANSAC band (inlier_thresh) is wide enough to
    # admit some head side-face points at oblique views, which tilt the fit, so
    # refine with a band that shrinks toward the measured noise level.
    inliers = best_inliers
    for _ in range(refine_iters):
        q = points[inliers]
        centroid = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - centroid, full_matrices=False)
        normal = vt[-1]
        dist = np.abs((points - centroid) @ normal)
        rms = float(np.sqrt(np.mean(dist[inliers] ** 2)))
        band = min(inlier_thresh, max(2.5 * rms, 0.25 * inlier_thresh))
        new_inliers = dist < band
        if new_inliers.sum() < 3 or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    q = points[inliers]
    centroid = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - centroid, full_matrices=False)
    normal = vt[-1]
    rms = float(np.sqrt(np.mean(((q - centroid) @ normal) ** 2)))
    # Report the inlier share against the RANSAC band so the ratio stays comparable
    support = np.abs((points - centroid) @ normal) < inlier_thresh
    return normal, centroid, inliers, rms, support


def frame_from_axis(z_axis):
    """Right-handed rotation matrix whose third column is z_axis. The x axis is
    the camera x axis projected perpendicular to z (roll is arbitrary)."""
    z = z_axis / np.linalg.norm(z_axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def quaternion_from_matrix(R):
    """(x, y, z, w) quaternion from a 3x3 rotation matrix."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return x, y, z, w


def estimate_bolt_pose(points, cfg, rng):
    """Classify the view and estimate center + axis for one bolt.

    cfg keys: max_fit_points, elongation_threshold, plane_inlier_thresh_m,
              ransac_iters, min_plane_inlier_ratio, head_height_m,
              shank_diameter_m, head_ratio_min, max_front_ratio
    Returns a dict with kind, center, axis (or None), quaternion, metrics.
    """
    if len(points) > cfg['max_fit_points']:
        points = points[rng.choice(len(points), cfg['max_fit_points'], replace=False)]

    surface_center = np.median(points, axis=0)
    toward_camera = -surface_center / np.linalg.norm(surface_center)
    mean, evals, evecs = principal_components(points)
    elongation = math.sqrt(evals[0] / max(evals[1], 1e-12))
    metrics = {'elongation': elongation, 'n_points': int(len(points))}

    def unknown(reason):
        metrics['reason'] = reason
        return {'kind': 'unknown', 'center': surface_center, 'surface_center': surface_center,
                'axis': None, 'quaternion': (0.0, 0.0, 0.0, 1.0), 'metrics': metrics}

    if elongation >= cfg['elongation_threshold']:
        # ---- SIDE-ON: bolt lying down, long axis visible ----
        axis = evecs[:, 0].copy()
        t = (points - mean) @ axis
        radial = np.linalg.norm((points - mean) - np.outer(t, axis), axis=1)
        lo, hi = np.percentile(t, [25, 75])
        end_pos, end_neg = radial[t >= hi], radial[t <= lo]
        if end_pos.size and end_neg.size:
            wide_pos, wide_neg = float(end_pos.mean()), float(end_neg.mean())
            head_ratio = max(wide_pos, wide_neg) / max(min(wide_pos, wide_neg), 1e-9)
            if wide_pos > wide_neg:
                axis = -axis                      # head at +t end: point axis toward the tip
            metrics['head_ratio'] = head_ratio
            metrics['axis_sign_reliable'] = head_ratio >= cfg['head_ratio_min']
        else:
            metrics['axis_sign_reliable'] = False

        # Surface normal facing the camera, perpendicular to the axis
        n = toward_camera - (toward_camera @ axis) * axis
        n = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-9 else toward_camera
        center = surface_center - n * (cfg['shank_diameter_m'] / 2.0)
        kind = 'side_on'
        metrics['tilt_deg'] = math.degrees(math.asin(min(1.0, abs(float(axis @ toward_camera)))))
    else:
        # ---- HEAD-ON: flat top of the head faces the camera ----
        fit = fit_plane_ransac(points, cfg['plane_inlier_thresh_m'], cfg['ransac_iters'], rng)
        if fit is None:
            return unknown('plane fit failed')
        normal, plane_point, inliers, rms, support = fit
        inlier_ratio = float(support.mean())
        metrics.update(plane_rms_mm=rms * 1000.0, inlier_ratio=inlier_ratio)
        if inlier_ratio < cfg['min_plane_inlier_ratio']:
            return unknown('no dominant head plane')
        if normal @ toward_camera < 0.0:
            normal = -normal
        # Nothing on a bolt sits in front of its head's top face. Many points in
        # front of the plane mean the fit landed on clutter or a noisy blob.
        front_ratio = float(np.mean((points - plane_point) @ normal > cfg['plane_inlier_thresh_m']))
        metrics['front_ratio'] = front_ratio
        if front_ratio > cfg['max_front_ratio']:
            return unknown('points in front of head plane')
        head_top = np.median(points[inliers], axis=0)
        head_top -= ((head_top - plane_point) @ normal) * normal   # snap onto the plane
        axis = -normal                                             # head -> tip, into the scene
        center = head_top - normal * (cfg['head_height_m'] / 2.0)
        kind = 'head_on'
        metrics['tilt_deg'] = math.degrees(math.acos(float(np.clip(normal @ toward_camera, -1.0, 1.0))))
        metrics['axis_sign_reliable'] = True

    return {'kind': kind, 'center': center, 'surface_center': surface_center,
            'axis': axis, 'quaternion': quaternion_from_matrix(frame_from_axis(axis)),
            'metrics': metrics}


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class BoltPoseEstimation(Node):
    def __init__(self):
        super().__init__('bolt_pose_estimation')

        # Front end
        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/k4a/rgb/camera_info')
        self.declare_parameter('model_path', 'best.pt')
        self.declare_parameter('confidence', 0.3)
        self.declare_parameter('erode_px', 2)
        self.declare_parameter('min_valid_pixels', 30)
        self.declare_parameter('mad_k', 3.0)
        self.declare_parameter('min_mad_m', 0.003)
        self.declare_parameter('use_tracking', True)
        self.declare_parameter('show_window', True)
        self.declare_parameter('log_period_s', 2.0)
        # Bolt geometry: SET THESE FOR YOUR BOLT SIZE (defaults are ISO 4017 M10)
        self.declare_parameter('head_height_m', 0.0064)
        self.declare_parameter('shank_diameter_m', 0.010)
        # Point-cloud analysis
        self.declare_parameter('elongation_threshold', 2.0)   # PCA std ratio separating side-on from head-on
        self.declare_parameter('plane_inlier_thresh_m', 0.004)
        self.declare_parameter('ransac_iters', 200)
        self.declare_parameter('min_plane_inlier_ratio', 0.4)
        self.declare_parameter('max_front_ratio', 0.1)        # max share of points in front of the head plane
        self.declare_parameter('head_ratio_min', 1.15)        # head/shank width ratio to trust axis direction
        self.declare_parameter('max_fit_points', 3000)
        self.declare_parameter('axis_marker_len_m', 0.04)
        self.declare_parameter('publish_cloud', True)

        gp = lambda name: self.get_parameter(name).value
        self.conf = float(gp('confidence'))
        self.min_valid_pixels = int(gp('min_valid_pixels'))
        self.mad_k = float(gp('mad_k'))
        self.min_mad_m = float(gp('min_mad_m'))
        self.use_tracking = bool(gp('use_tracking'))
        self.show_window = bool(gp('show_window'))
        self.log_period_s = float(gp('log_period_s'))
        self.axis_len = float(gp('axis_marker_len_m'))
        self.publish_cloud = bool(gp('publish_cloud'))
        self.cfg = {
            'head_height_m': float(gp('head_height_m')),
            'shank_diameter_m': float(gp('shank_diameter_m')),
            'elongation_threshold': float(gp('elongation_threshold')),
            'plane_inlier_thresh_m': float(gp('plane_inlier_thresh_m')),
            'ransac_iters': int(gp('ransac_iters')),
            'min_plane_inlier_ratio': float(gp('min_plane_inlier_ratio')),
            'max_front_ratio': float(gp('max_front_ratio')),
            'head_ratio_min': float(gp('head_ratio_min')),
            'max_fit_points': int(gp('max_fit_points')),
        }

        erode_px = int(gp('erode_px'))
        self.kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (2 * erode_px + 1, 2 * erode_px + 1))
                       if erode_px > 0 else None)

        self.rng = np.random.default_rng(0)
        self.bridge = CvBridge()
        self.model = YOLO(gp('model_path'))

        self.K = None
        self.D = None
        self.ray_x = None
        self.ray_y = None
        self.last_log_time = self.get_clock().now()

        # Sensor-data QoS (best effort) is compatible with both reliable and
        # best-effort publishers, so it connects whatever the camera driver uses.
        self.rgb_topic = gp('rgb_topic')
        self.depth_topic = gp('depth_topic')
        self.camera_info_topic = gp('camera_info_topic')
        qos = qos_profile_sensor_data
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                                queue_size=10, slop=0.1)
        self.sync.registerCallback(self.kinect_callback)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, qos)

        # Until calibration arrives, report every few seconds what the node can see
        self.frames_received = 0
        self.wait_timer = self.create_timer(3.0, self.report_waiting)

        self.pose_pub = self.create_publisher(PoseArray, '/hex_bolt/poses', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/hex_bolt/pose_markers', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/hex_bolt/points', 10)

        self.get_logger().info('Bolt pose estimator started. Waiting for camera info...')

    def report_waiting(self):
        """Diagnostics while no CameraInfo has arrived: publishers, QoS, visible topics."""
        if self.K is not None:
            self.wait_timer.cancel()
            return
        log = self.get_logger()
        try:
            pubs = self.get_publishers_info_by_topic(self.camera_info_topic)
            if pubs:
                rel = sorted({getattr(p.qos_profile.reliability, 'name', str(p.qos_profile.reliability))
                              for p in pubs})
                log.warn(f'No CameraInfo received yet on "{self.camera_info_topic}" although '
                         f'{len(pubs)} publisher(s) exist (reliability: {", ".join(rel)}). '
                         'The driver may not be streaming; check it with `ros2 topic hz`.')
            else:
                visible = sorted(name for name, types in self.get_topic_names_and_types()
                                 if 'sensor_msgs/msg/CameraInfo' in types)
                log.warn(f'No publisher on "{self.camera_info_topic}". CameraInfo topics visible: '
                         f'{visible if visible else "none (is the driver running, same ROS_DOMAIN_ID?)"}. '
                         'Select one with: --ros-args -p camera_info_topic:=<topic> '
                         '(and set rgb_topic / depth_topic to match).')
            log.info(f'Synced RGB+depth pairs received: {self.frames_received} '
                     f'({self.rgb_topic}: {self.count_publishers(self.rgb_topic)} publisher(s), '
                     f'{self.depth_topic}: {self.count_publishers(self.depth_topic)} publisher(s))')
        except Exception as e:
            log.warn(f'Still waiting for CameraInfo on "{self.camera_info_topic}" ({e})')

    def camera_info_callback(self, msg):
        if self.K is not None:
            return
        if msg.width == 0 or msg.k[0] == 0.0:
            self.get_logger().warn('CameraInfo received but it is uncalibrated (fx = 0 or width = 0); '
                                   'waiting for a valid one.', throttle_duration_sec=5.0)
            return
        self.get_logger().info(
            f'CameraInfo received on "{self.camera_info_topic}" ({msg.width}x{msg.height}, '
            f'{msg.distortion_model or "no distortion model"}). Building ray table...')
        if msg.distortion_model not in ('', 'plumb_bob', 'rational_polynomial'):
            self.get_logger().warn(
                f'Distortion model "{msg.distortion_model}" is not handled by '
                'cv2.undistortPoints; rays may be inaccurate.')
        try:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
            t0 = time.monotonic()
            ray_x, ray_y = build_ray_table(K, D, msg.width, msg.height)
        except Exception:
            self.get_logger().error('Could not build the ray table from CameraInfo:\n'
                                    + traceback.format_exc(), throttle_duration_sec=5.0)
            return
        self.D, self.ray_x, self.ray_y = D, ray_x, ray_y
        self.K = K   # set last: marks calibration as ready for the frame callback
        self.get_logger().info(
            f'Intrinsics locked: fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} '
            f'cy={K[1, 2]:.2f}; ray table built in {time.monotonic() - t0:.2f} s.')

    def kinect_callback(self, rgb_msg, depth_msg):
        self.frames_received += 1
        if self.K is None:
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_m = depth_to_meters(depth_raw, depth_msg.encoding)

            h, w = rgb.shape[:2]
            if depth_m.shape != (h, w) or self.ray_x.shape != (h, w):
                self.get_logger().error(
                    f'Size mismatch: RGB {(h, w)}, depth {depth_m.shape}, '
                    f'camera_info {self.ray_x.shape}.', throttle_duration_sec=5.0)
                return

            bolts = self.detect(rgb, depth_m)
            self.publish(rgb_msg.header, bolts)
            self.log_bolts(bolts)

            if self.show_window:
                self.draw(rgb, bolts)
                cv2.imshow('Bolt pose estimation', rgb)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')

    def detect(self, rgb, depth_m):
        kwargs = dict(conf=self.conf, retina_masks=True, verbose=False)
        res = (self.model.track(rgb, persist=True, **kwargs) if self.use_tracking
               else self.model.predict(rgb, **kwargs))[0]
        if res.masks is None or len(res.masks) == 0:
            return []

        masks = res.masks.data.cpu().numpy()
        ids = (res.boxes.id.int().cpu().numpy().tolist()
               if getattr(res.boxes, 'id', None) is not None else None)

        bolts = []
        for i, mask_prob in enumerate(masks):
            raw_mask, mask = prepare_mask(mask_prob, depth_m.shape, self.kernel)
            pts = robust_surface_points(mask, depth_m, self.ray_x, self.ray_y,
                                        self.mad_k, self.min_mad_m, self.min_valid_pixels)
            if pts is None:
                continue
            est = estimate_bolt_pose(pts, self.cfg, self.rng)
            est.update(id=int(ids[i]) if ids else i + 1, points=pts, raw_mask=raw_mask)
            bolts.append(est)
        return bolts

    # -- output ----------------------------------------------------------------

    def publish(self, header, bolts):
        poses = PoseArray()
        poses.header = header
        markers = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()

        for b in bolts:
            cx, cy, cz = (float(c) for c in b['center'])
            qx, qy, qz, qw = (float(q) for q in b['quaternion'])

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = cx, cy, cz
            pose.orientation.x, pose.orientation.y = qx, qy
            pose.orientation.z, pose.orientation.w = qz, qw
            poses.poses.append(pose)

            if b['axis'] is not None:
                arrow = Marker()
                arrow.header = header
                arrow.ns = 'hex_bolt_axis'
                arrow.id = b['id']
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                tip = b['center'] + b['axis'] * self.axis_len
                arrow.points = [Point(x=cx, y=cy, z=cz),
                                Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2]))]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.004, 0.008, 0.01
                reliable = b['metrics'].get('axis_sign_reliable', False)
                arrow.color.r, arrow.color.g, arrow.color.b = (0.1, 0.9, 0.1) if reliable else (1.0, 0.6, 0.0)
                arrow.color.a = 1.0
                arrow.lifetime = lifetime
                markers.markers.append(arrow)

            label = Marker()
            label.header = header
            label.ns = 'hex_bolt_label'
            label.id = b['id']
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y, label.pose.position.z = cx, cy - 0.03, cz
            label.pose.orientation.w = 1.0
            label.scale.z = 0.02
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f'#{b["id"]} {b["kind"]}'
            label.lifetime = lifetime
            markers.markers.append(label)

        self.pose_pub.publish(poses)
        if markers.markers:
            self.marker_pub.publish(markers)

        if self.publish_cloud and bolts:
            rows = np.vstack([np.column_stack([b['points'], np.full(len(b['points']), b['id'])])
                              for b in bolts]).astype(np.float32)
            fields = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1)
                      for k, n in enumerate(('x', 'y', 'z', 'bolt_id'))]
            self.cloud_pub.publish(point_cloud2.create_cloud(header, fields, rows))

    def log_bolts(self, bolts):
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds / 1e9 < self.log_period_s:
            return
        self.last_log_time = now
        log = self.get_logger()
        log.info('-' * 85)
        if not bolts:
            log.info('No bolts localized.')
        for b in sorted(bolts, key=lambda d: d['id']):
            m = b['metrics']
            c = b['center']
            line = (f'Bolt #{b["id"]} [{b["kind"]}]: center=({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}) m, '
                    f'elong={m["elongation"]:.2f}, n={m["n_points"]}')
            if b['axis'] is not None:
                a = b['axis']
                line += (f', axis=({a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}), '
                         f'tilt={m["tilt_deg"]:.1f} deg')
                if not m.get('axis_sign_reliable', False):
                    line += ' (head/tip direction uncertain)'
            if 'plane_rms_mm' in m:
                line += f', plane_rms={m["plane_rms_mm"]:.2f} mm, inliers={100 * m["inlier_ratio"]:.0f}%'
            if 'reason' in m:
                line += f' ({m["reason"]})'
            log.info(line)

    def project(self, point):
        uv, _ = cv2.projectPoints(np.asarray(point, dtype=np.float64).reshape(1, 1, 3),
                                  np.zeros(3), np.zeros(3), self.K, self.D)
        return int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))

    def draw(self, image, bolts):
        overlay = image.copy()
        for b in bolts:
            overlay[b['raw_mask']] = (0, 200, 0)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
        for b in bolts:
            u, v = self.project(b['center'])
            cv2.circle(image, (u, v), 4, (0, 0, 255), -1)
            if b['axis'] is not None:
                tu, tv = self.project(b['center'] + b['axis'] * self.axis_len)
                color = (0, 255, 0) if b['metrics'].get('axis_sign_reliable') else (0, 165, 255)
                cv2.arrowedLine(image, (u, v), (tu, tv), color, 2, tipLength=0.25)
            cv2.putText(image, f'#{b["id"]} {b["kind"]} {np.linalg.norm(b["center"]):.3f} m',
                        (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main(args=None):
    rclpy.init(args=args)
    node = BoltPoseEstimation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
