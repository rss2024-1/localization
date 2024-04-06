from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy

from sensor_msgs.msg import PointCloud, LaserScan
from geometry_msgs.msg import Point32

assert rclpy
import numpy as np


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
        self.particle_filter_publisher = self.create_publisher()

        self.probabilities = None
        # Number of particles
        N = 200

        # Initialize x0 and y0 with noise around 0
        x0 = np.random.normal(loc=0.0, scale=0.001, size=N)
        y0 = np.random.normal(loc=0.0, scale=0.001, size=N)

        # Initialize theta0 to 0
        theta0 = np.zeros(N)

        # Combine x0, y0, and theta0 into a N*3 array
        self.particles = np.column_stack((x0, y0, theta0))
        ### 1. Initialize a bunch of particles, with some noise (__init__ function)
        ### Include a way to visualize the particles too (__init__ function)

        ### While loop (so basicaly as the roobt runs):
        ###     (in no particular order)
        ####    2. Make updates to the position via calling motion model. Via motion_model in odom_callback.
        ####    3. Make updates to our probability distribution of particles. Via sensor_model in laser_callback.
        ####    finally, pose_callback is designed to just publish what our point cloud is


    def laser_callback(self, data):
        ### sensor_model + "survival of fittest" on the particles
        probabilities = self.sensor_model.evaluate(self.particles, data)
        self.probabilities = probabilities
        # Generate a list of indices to select from self.particles
        indices = np.arange(len(self.particles))

        # Randomly sample indices based on probabilities
        sampled_indices = np.random.choice(indices, size=len(self.particles), p=probabilities)

        # Create the new particles array
        self.particles = self.particles[sampled_indices]


    def pose_callback(self, data):
        # Compute the weighted average of the particles
        ### compute average particle pose, publish, weight on probabilities
        weighted_particles = np.average(self.particles, weights=self.probabilities, axis=0)
        x_avg = weighted_particles[0]
        y_avg = weighted_particles[1]
        theta_avg = weighted_particles[2]

        # Create an Odometry message
        odom_msg = Odometry()

        # Fill in the message fields
        odom_msg.pose.pose.position.x = x_avg
        odom_msg.pose.pose.position.y = y_avg
        odom_msg.pose.pose.orientation.z = np.sin(theta_avg / 2.0)
        odom_msg.pose.pose.orientation.w = np.cos(theta_avg / 2.0)

        # Set the frame_id
        odom_msg.header.frame_id = "/map"

        # Publish the message
        self.odom_pub.publish(odom_msg)


    def odom_callback(self, data):
        ### motion_model
        self.particles = self.motion_model.evaluate(self.particles, data)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
