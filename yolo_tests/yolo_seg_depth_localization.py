#!/usr/bin/env python3
"""
yolo_seg_depth_localization.py

3D hex bolt localization from a YOLO instance-segmentation model and an
Azure Kinect registered depth image (depth_to_rgb).

Per frame:
  1. YOLO-seg inference with full-resolution masks (retina_masks=True) and,
     optionally, tracking so each bolt keeps the same ID across frames.
  2. Each mask is eroded to discard boundary pixels and ToF "flying pixels".
  3. Every valid mask pixel is back-projected to 3D through a precomputed table
     of undistorted pixel rays (lens distortion is handled once, at startup).
  4. Depth outliers (specular dropouts, multipath spikes) are rejected with a
     median-absolute-deviation (MAD) test.
  5. The per-axis median of the surviving 3D points is reported.

The reported point is the center of the bolt's VISIBLE SURFACE, not its
volumetric center. bolt_pose_estimation.py adds orientation and the
surface-to-center offset.

Output frame: the RGB camera optical frame of the incoming images
(x right, y down, z forward), in meters.

Topics
  sub  rgb_topic          sensor_msgs/Image       (bgr8)
  sub  depth_topic        sensor_msgs/Image       (16UC1 mm or 32FC1 m, registered to RGB)
  sub  camera_info_topic  sensor_msgs/CameraInfo  (must match the RGB image size)
  pub  /hex_bolt/spatial_coordinate  geometry_msgs/PointStamped  (one message per bolt)
  pub  /hex_bolt/markers             visualization_msgs/MarkerArray (sphere + ID label per bolt)
"""

import math
import time
import traceback

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray


# ---------------------------------------------------------------------------
# Geometry helpers (no ROS dependencies, so they can be unit-tested directly)
# ---------------------------------------------------------------------------

def build_ray_table(K, D, width, height):
    """Return (ray_x, ray_y): the undistorted normalized ray (X/Z, Y/Z) for every
    pixel. Back-projection is then just X = ray_x * Z, Y = ray_y * Z."""
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
    """Convert a depth image to float32 meters. Invalid pixels become 0."""
    if encoding in ('16UC1', 'mono16'):
        return depth_image.astype(np.float32) * 0.001
    if encoding == '32FC1':
        d = depth_image.astype(np.float32, copy=False)
        return np.where(np.isfinite(d), d, 0.0).astype(np.float32)
    raise ValueError(f'Unsupported depth encoding "{encoding}" (expected 16UC1 or 32FC1)')


def prepare_mask(mask_prob, image_shape, kernel):
    """Binarize a YOLO mask, match it to the image size, and erode it.
    Returns (raw_mask, eroded_mask) as boolean arrays."""
    raw = (mask_prob > 0.5).astype(np.uint8)
    if raw.shape != image_shape:
        raw = cv2.resize(raw, (image_shape[1], image_shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    eroded = cv2.erode(raw, kernel) if kernel is not None else raw
    return raw.astype(bool), eroded.astype(bool)


def robust_surface_points(mask, depth_m, ray_x, ray_y,
                          mad_k=3.0, min_mad_m=0.003, min_pixels=30):
    """Back-project the valid pixels under `mask` to an (N, 3) array of
    camera-frame points, after rejecting depth outliers with a MAD test.
    Returns None if too few valid pixels remain."""
    sel = mask & (depth_m > 0.0)
    if int(sel.sum()) < min_pixels:
        return None
    z = depth_m[sel]
    med = float(np.median(z))
    # 1.4826 * MAD estimates the standard deviation for Gaussian noise. The floor
    # stops quantized (1 mm) depth with near-zero spread from rejecting everything.
    sigma = max(1.4826 * float(np.median(np.abs(z - med))), min_mad_m)
    keep = np.abs(z - med) <= mad_k * sigma
    if int(keep.sum()) < min_pixels:
        return None
    z = z[keep]
    x = ray_x[sel][keep] * z
    y = ray_y[sel][keep] * z
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class YoloSegDepthLocalization(Node):
    def __init__(self):
        super().__init__('yolo_seg_depth_localization')

        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/k4a/rgb/camera_info')
        self.declare_parameter('model_path', 'best.pt')
        self.declare_parameter('confidence', 0.1)
        self.declare_parameter('erode_px', 2)            # mask erosion radius (pixels)
        self.declare_parameter('min_valid_pixels', 30)   # skip detections with fewer valid depth pixels
        self.declare_parameter('mad_k', 3.0)             # outlier cutoff, in robust standard deviations
        self.declare_parameter('min_mad_m', 0.003)       # floor on the robust std (meters)
        self.declare_parameter('use_tracking', True)     # stable IDs across frames via model.track()
        self.declare_parameter('show_window', True)
        self.declare_parameter('log_period_s', 2.0)

        gp = lambda name: self.get_parameter(name).value
        self.conf = float(gp('confidence'))
        self.min_valid_pixels = int(gp('min_valid_pixels'))
        self.mad_k = float(gp('mad_k'))
        self.min_mad_m = float(gp('min_mad_m'))
        self.use_tracking = bool(gp('use_tracking'))
        self.show_window = bool(gp('show_window'))
        self.log_period_s = float(gp('log_period_s'))

        erode_px = int(gp('erode_px'))
        self.kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (2 * erode_px + 1, 2 * erode_px + 1))
                       if erode_px > 0 else None)

        self.bridge = CvBridge()
        self.model = YOLO(gp('model_path'))

        # Calibration (filled from the first CameraInfo message)
        self.K = None
        self.D = None
        self.ray_x = None
        self.ray_y = None

        self.last_log_time = self.get_clock().now()
        self.depth_encoding_logged = False

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

        self.point_pub = self.create_publisher(PointStamped, '/hex_bolt/spatial_coordinate', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/hex_bolt/markers', 10)

        self.get_logger().info('YOLO-seg 3D bolt localizer started. Waiting for camera info...')

    # -- calibration -----------------------------------------------------------

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

    # -- main pipeline ---------------------------------------------------------

    def kinect_callback(self, rgb_msg, depth_msg):
        self.frames_received += 1
        if self.K is None:
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_m = depth_to_meters(depth_raw, depth_msg.encoding)
            if not self.depth_encoding_logged:
                self.get_logger().info(f'Depth encoding: {depth_msg.encoding} (converted to meters)')
                self.depth_encoding_logged = True

            h, w = rgb.shape[:2]
            if depth_m.shape != (h, w):
                self.get_logger().error(
                    f'Depth {depth_m.shape} and RGB {(h, w)} sizes differ; '
                    'use the depth_to_rgb topic.', throttle_duration_sec=5.0)
                return
            if self.ray_x.shape != (h, w):
                self.get_logger().error(
                    f'CameraInfo size {self.ray_x.shape[::-1]} does not match the RGB '
                    f'image {(w, h)}; intrinsics would be wrong.', throttle_duration_sec=5.0)
                return

            detections = self.detect(rgb, depth_m)
            self.publish(rgb_msg.header, detections)
            self.log_detections(detections)

            if self.show_window:
                self.draw(rgb, detections)
                cv2.imshow('YOLO-seg 3D bolt localization', rgb)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')

    def infer(self, image):
        kwargs = dict(conf=self.conf, retina_masks=True, verbose=False)
        if self.use_tracking:
            return self.model.track(image, persist=True, **kwargs)[0]
        return self.model.predict(image, **kwargs)[0]

    def detect(self, rgb, depth_m):
        """Run segmentation and return one dict per successfully localized bolt."""
        res = self.infer(rgb)
        if res.masks is None or len(res.masks) == 0:
            return []

        masks = res.masks.data.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy().tolist()
        ids = (res.boxes.id.int().cpu().numpy().tolist()
               if getattr(res.boxes, 'id', None) is not None else None)

        detections = []
        for i, mask_prob in enumerate(masks):
            raw_mask, mask = prepare_mask(mask_prob, depth_m.shape, self.kernel)
            pts = robust_surface_points(mask, depth_m, self.ray_x, self.ray_y,
                                        self.mad_k, self.min_mad_m, self.min_valid_pixels)
            if pts is None:
                continue
            detections.append({
                'id': int(ids[i]) if ids else i + 1,   # 1-based when not tracking
                'conf': float(confs[i]),
                'center': np.median(pts, axis=0),       # visible-surface center (m)
                'n_points': int(len(pts)),
                'raw_mask': raw_mask,
                'mask': mask,
            })
        return detections

    # -- output ----------------------------------------------------------------

    def publish(self, header, detections):
        markers = MarkerArray()
        lifetime = Duration(seconds=0.5).to_msg()
        for det in detections:
            x, y, z = (float(c) for c in det['center'])

            msg = PointStamped()
            msg.header = header
            msg.point.x, msg.point.y, msg.point.z = x, y, z
            self.point_pub.publish(msg)

            sphere = Marker()
            sphere.header = header
            sphere.ns = 'hex_bolt'
            sphere.id = det['id']
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.015
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.1, 0.9, 0.1, 0.9
            sphere.lifetime = lifetime
            markers.markers.append(sphere)

            label = Marker()
            label.header = header
            label.ns = 'hex_bolt_label'
            label.id = det['id']
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y - 0.025, z  # y is down
            label.pose.orientation.w = 1.0
            label.scale.z = 0.02
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f'#{det["id"]}'
            label.lifetime = lifetime
            markers.markers.append(label)

        if markers.markers:
            self.marker_pub.publish(markers)

    def log_detections(self, detections):
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds / 1e9 < self.log_period_s:
            return
        self.last_log_time = now
        self.get_logger().info('-' * 85)
        if not detections:
            self.get_logger().info('No bolts localized.')
        for det in sorted(detections, key=lambda d: d['id']):
            x, y, z = det['center']
            self.get_logger().info(
                f'Bolt #{det["id"]}: X={x:+.4f} Y={y:+.4f} Z={z:+.4f} m '
                f'(range {math.sqrt(x * x + y * y + z * z):.3f} m, '
                f'{det["n_points"]} px, conf {det["conf"]:.2f})')

    def project(self, point):
        """Project a camera-frame 3D point to integer pixel coordinates (with distortion)."""
        uv, _ = cv2.projectPoints(np.asarray(point, dtype=np.float64).reshape(1, 1, 3),
                                  np.zeros(3), np.zeros(3), self.K, self.D)
        u, v = uv[0, 0]
        return int(round(u)), int(round(v))

    def draw(self, image, detections):
        # Translucent fill = raw mask; outline = eroded mask actually used for depth
        overlay = image.copy()
        for det in detections:
            overlay[det['raw_mask']] = (0, 200, 0)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
        for det in detections:
            contours, _ = cv2.findContours(det['mask'].astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            u, v = self.project(det['center'])
            cv2.circle(image, (u, v), 4, (0, 0, 255), -1)
            rng = float(np.linalg.norm(det['center']))
            cv2.putText(image, f'#{det["id"]} {rng:.3f} m', (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegDepthLocalization()
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
