from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from rclpy.time import Time

from nav_msgs.msg import Odometry
import geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
import tf_transformations

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
    
    def laser_callback(self, scan): 
        probabilities = self.sensor_model.evaluate(self.particles, scan.intensities)
        resampled_indices = np.random.choice(self.particles.shape[0], size=self.particles.shape[0], replace=True, p=probabilities)
        resampled_particles = self.particles[resampled_indices]
        self.particles = resampled_particles
        self.publish_transform(self.particles)


    def odom_callback(self, odometry): 
        dx = odometry.twist.twist.linear.x
        dy = odometry.twist.twist.linear.y
        dth = odometry.twist.twist.angular.z
        od = [dx, dy, dth]
        updated_particles = self.motion_model.evaluate(self.particles, od)
        
        # return updated_particles
        self.particles = updated_particles
        self.publish_transform(self.particles)
    
    # def pose_callback(self, msg): 
    #     """
    #         var:
    #             msg = PoseWithCovariance Message, vars: pose, covariance
    #                 pose = Pose Message, vars: Point position, Quaternion orientation
    #     """
    #     #init particles
    #     # np.random.choice: (original array size, desired sample size, probabilities list of original array)
    #     # self.particles = np.random.choice(self.particles, )
    #     x = msg.pose.pose.position.x
    #     y = msg.pose.pose.position.y
    #     z = msg.pose.pose.position.z

    #     q_x = msg.pose.pose.orientation.x
    #     q_y = msg.pose.pose.orientation.y
    #     q_z = msg.pose.pose.orientation.z
    #     q_w = msg.pose.pose.orientation.w


    #     self.particles = np.array(pose) + np.random.normal(loc=0.0, scale = .001, size=(len(self.particles),3))
    #     self.get_logger().info("init particles from pose")
        
    def pose_callback(self, msg): 
        """
        var:
            msg = PoseWithCovarianceStamped Message, vars: pose, covariance
                pose = Pose Message, vars: Point position, Quaternion orientation
        """
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        th = 2*np.arccos(msg.pose.pose.orientation.w)

        # Update particles with the new position and orientation
        newx = x + np.random.normal(loc=0.0, scale=0.001, size=(len(self.particles), 2))
        newy = y +  np.random.normal(loc=0.0, scale=0.001, size=(len(self.particles), 2))
        newth = np.angle(np.exp(1j * (th + np.random.default_rng().uniform(low=0.0, high=2*np.pi, size=len(self.particles)))))

        self.particles = np.array([np.array([x,y,th]) for x,y,th in zip(newx, newy, newth)])        
        
        self.get_logger().info("Initialized particles from pose")

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
        odom_pub_msg.pose.pose.orientation.w = avg_angle
        self.odom_pub.publish(odom_pub_msg)

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

    def se3_to_tf(self, mat: np.ndarray, time: Time, parent: str, child: str) -> TransformStamped:
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
