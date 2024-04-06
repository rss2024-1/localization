from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy
from sensor_msgs.msg import LaserScan

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def laser_callback(self, scan): 
        probabilities = self.sensor_model.evaluate(self.particles, scan)
        resampled_particles = np.random.choice(self.particles, p=probabilities)
        self.particles = resampled_particles
        self.publish_transform(self.particles)

    def odom_callback(self, odometry): 
        updated_particles = self.motion_model.evaluate(self.particles, odometry)
        
        # return updated_particles
        self.particles = updated_particles
        self.publish_transform(self.particles)
    
    def pose_callback(self, pose): 
        #init particles
        self.particles = pose + np.random.normal(loc=0.0, scale = .001, size=(len(self.particles),3))
        self.get_logger().info("init particles from pose")
    
    def publish_transform(self, particles): 
        #average particle pose 
        sin_sum = np.sum(np.sin(self.particles[:,2]))
        cos_sum = np.sum(np.cos(self.particles[:,2]))
        avg_angle = np.atan2(sin_sum, cos_sum)

        avg_x = np.avg(self.particles[:,0])
        avg_y = np.avg(self.particles[:,1])
        
        particle_pose = [avg_x, avg_y, avg_angle]
        
        #step 2: convert robot transform to 4x4 np array 
        # robot_to_world: np.ndarray = self.tf_to_se3(tf_robot_to_world.transform)

        # #step 3: compute current transform of left camera wrt world
        # left_cam_tf = robot_to_world @ self.left_cam

        # #step 4: compute transform of right camera wrt left 
        # right_cam_tf = left_cam_tf @ np.linalg.inv(self.right_cam)
        
        # #broadcast transforms for cameras to TF tree
        # now = self.get_clock().now()
        # left_cam_tf_msg = self.se3_to_tf(left_cam_tf, now, parent='world', child='left_camera')
        # self.br.sendTransform([tf_robot_to_world, left_cam_tf_msg])

        # right_cam_tf_msg = self.se3_to_tf(right_cam_tf, now, parent='world', child='right_camera')
        # self.br.sendTransform([tf_robot_to_world, right_cam_tf_msg])
        
        #publish transform between /map frame and frame for exp car base link
        tf_map_baselink = self.sensor_model.map @ particle_pose
        now = self.get_clock().now()
        out = self.se3_to_tf(tf_map_baselink, now, parent='map', child='base_link')
        self.odom_pub(out)

    def tf_to_se3(self, transform: TransformStamped.transform) -> np.ndarray:
        """
        Convert a TransformStamped message to a 4x4 SE3 matrix 
        """
        q = transform.rotation
        q = [q.x, q.y, q.z, q.w]
        t = transform.translation
        mat = tf_transformations.quaternion_matrix(q)
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        return mat

    def se3_to_tf(self, mat: np.ndarray, time: Any, parent: str, child: str) -> TransformStamped:
        """
        Convert a 4x4 SE3 matrix to a TransformStamped message
        """
        obj = geometry_msgs.msg.TransformStamped()

        # current time
        obj.header.stamp = time.to_msg()

        # frame names
        obj.header.frame_id = parent
        obj.child_frame_id = child

        # translation component
        obj.transform.translation.x = mat[0, 3]
        obj.transform.translation.y = mat[1, 3]
        obj.transform.translation.z = mat[2, 3]

        # rotation (quaternion)
        q = tf_transformations.quaternion_from_matrix(mat)
        obj.transform.rotation.x = q[0]
        obj.transform.rotation.y = q[1]
        obj.transform.rotation.z = q[2]
        obj.transform.rotation.w = q[3]

        return obj



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
