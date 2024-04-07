from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from rclpy.time import Time

from nav_msgs.msg import Odometry
import geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseArray, Pose, Quaternion
import tf_transformations as tf

from rclpy.node import Node
import rclpy
from sensor_msgs.msg import LaserScan

assert rclpy
import numpy as np


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.particles = np.zeros((100,3))

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

        ## personal notes:
        ## need to get odom reading (which is a twist), convert it 

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

        self.particles_pub = self.create_publisher(PoseArray, '/posearray', 1)
        
        self.frame = '/map'
        # self.frame = '/map'
        self.curr_time = self.get_clock().now()

    def laser_callback(self, scan): 
        probabilities = self.sensor_model.evaluate(self.particles, scan.intensities)
        try: 
            probabilities /= sum(probabilities)
        except: 
            probabilities = np.empty(100)
            probabilities.fill(1/100) 
        resampled_indices = np.random.choice(self.particles.shape[0], size=self.particles.shape[0], replace=True, p=probabilities)
        resampled_particles = self.particles[resampled_indices]
        self.particles = resampled_particles

        self.publish_transform(self.particles)


    def odom_callback(self, odometry): 
        dt = self.get_clock().now() - self.curr_time
        self.curr_time = self.get_clock().now()

        dx = odometry.twist.twist.linear.x * dt
        dy = odometry.twist.twist.linear.y * dt
        dth = odometry.twist.twist.angular.z * dt
        od = [dx, dy, dth]
        updated_particles = self.motion_model.evaluate(self.particles, od)
        
        # return updated_particles
        self.particles = updated_particles
        self.publish_transform(self.particles)
        
    def pose_callback(self, msg): 
        """
        var:
            msg = PoseWithCovarianceStamped Message, vars: pose, covariance
                pose = Pose Message, vars: Point position, Quaternion orientation
        """
        # # Extract position
        # x = msg.pose.pose.position.x
        # y = msg.pose.pose.position.y
        # # euler fr/ q
        # odom_quat = tf.euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        # self.get_logger().info("in pose callback")
        # self.get_logger().info(f"X: {x}")
        # self.get_logger().info(f"y: {y}")
        # self.get_logger().info(f"odom_quat is: {odom_quat}")
        # th = 0# odom_quat[2]

        
        # od = [x, y, th]
        # # updated_particles = self.motion_model.evaluate(self.particles, od)
        # self.particles[:,0] = x
        # self.particles[:,1]=y
        
        # return updated_particles
        # self.particles = updated_particles
        self.publish_transform(self.particles)


    def publish_transform(self, particles): 
        #average particle pose 
        sin_sum = np.sum(np.sin(particles[:,2]))
        cos_sum = np.sum(np.cos(particles[:,2]))
        avg_angle = np.arctan2(sin_sum, cos_sum)

        avg_x = np.average(particles[:,0])
        avg_y = np.average(particles[:,1])
        
        # particle_pose = [avg_x, avg_y, avg_angle]
        
        #publish transform between /map frame and frame for exp car base link
        # tf_map_baselink = self.sensor_model.map @ particle_pose
        # now = self.get_clock().now()
        # out = self.se3_to_tf(tf_map_baselink, now, parent='map', child='base_link')
        
        odom_pub_msg = Odometry() # creating message template in case ros is mad
        odom_pub_msg.pose.pose.position.x = avg_x #out # specifically ONLY publish to the pose variable
        odom_pub_msg.pose.pose.position.y = avg_y
        odom_pub_msg.pose.pose.position.z = 0.0
        odom_quat = tf.quaternion_from_euler(0, 0, avg_angle)
        self.get_logger().info(f"avg X: {avg_x}")
        self.get_logger().info(f"avg y: {avg_y}")
        self.get_logger().info(f"odom quat: {odom_quat}")
        odom_pub_msg.pose.pose.orientation = Quaternion(x=odom_quat[0], y=odom_quat[1], z=odom_quat[2], w=odom_quat[3])
        #header

        odom_pub_msg.header.stamp = rclpy.time.Time().to_msg()
        odom_pub_msg.header.frame_id = "/map"
        self.odom_pub.publish(odom_pub_msg)

        #posearray
        particles_msg = PoseArray()
        particles_msg.header.stamp = rclpy.time.Time().to_msg()
        particles_msg.header.frame_id = "/map"
        poses = []
        for x,y,th in self.particles:
            pose_msg = Pose() 
            pose_msg.position.x = x
            pose_msg.position.y = y
            pose_msg.position.z = 0.0
            quat = tf.quaternion_from_euler(0, 0, th)
            pose_msg.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            poses.append(pose_msg)
        particles_msg.poses = poses
        self.particles_pub.publish(particles_msg)


    # def tf_to_se3(self, transform: TransformStamped.transform) -> np.ndarray:
    #     """
    #     Convert a TransformStamped message to a 4x4 SE3 matrix 
    #     """
    #     q = transform.rotation
    #     q = [q.x, q.y, q.z, q.w]
    #     t = transform.translation
    #     mat = tf.quaternion_matrix(q)
    #     mat[0, 3] = t.x
    #     mat[1, 3] = t.y
    #     mat[2, 3] = t.z
    #     return mat

    # def se3_to_tf(self, mat: np.ndarray, time: Time, parent: str, child: str) -> TransformStamped:
    #     """
    #     Convert a 4x4 SE3 matrix to a TransformStamped message
    #     """
    #     obj = geometry_msgs.msg.TransformStamped()

    #     # current time
    #     obj.header.stamp = time.to_msg()

    #     # frame names
    #     obj.header.frame_id = parent
    #     obj.child_frame_id = child

    #     # translation component
    #     obj.transform.translation.x = mat[0, 3]
    #     obj.transform.translation.y = mat[1, 3]
    #     obj.transform.translation.z = mat[2, 3]

    #     # rotation (quaternion)
    #     q = tf.quaternion_from_matrix(mat)
    #     obj.transform.rotation.x = q[0]
    #     obj.transform.rotation.y = q[1]
    #     obj.transform.rotation.z = q[2]
    #     obj.transform.rotation.w = q[3]

    #     return obj



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
