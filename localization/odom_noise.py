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
import threading


class OdomNoise(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.particles = np.zeros((200,3))
        self.mutex = threading.Lock()

        self.declare_parameter('noise_level', "default")
        self.noise_level = self.get_parameter('noise_level').get_parameter_value().value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)


        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/noisy_odom", 1)


        self.frame = '/map'

        self.curr_time = self.get_clock().now()


    def odom_callback(self, odometry): 
        self.mutex.acquire(blocking=True)
        dt = (self.get_clock().now() - self.curr_time).nanoseconds * 1e-9
        self.curr_time = self.get_clock().now()

        dx = odometry.twist.twist.linear.x * dt
        dy = odometry.twist.twist.linear.y * dt
        dth = odometry.twist.twist.angular.z * dt

        #add gaussian noise to velocity and angular velocity
        dx = np.random.normal(loc=dx, scale=self.noise_level)
        dy = np.random.normal(loc=dy, scale=self.noise_level)
        dth = np.random.normal(loc=dth, scale=self.noise_level)



        od = [dx, dy, dth]

        # # return updated_particles
        # self.publish_transform(self.particles)
        # self.mutex.release()
        odom_pub_msg = odometry 
        odom_pub_msg.twist.twist.linear.x = dx/dt
        odom_pub_msg.twist.twist.linear.y = dy/dt
        odom_pub_msg.twist.twist.angular.z = dth/dt

        # odom_pub_msg = Odometry() # creating message template in case ros is mad
        # odom_pub_msg.pose.pose.position.x = odometry.pose.pose.position.x
        # odom_pub_msg.pose.pose.position.y = odometry.pose.pose.position.y
        # odom_pub_msg.pose.pose.position.z = odometry.pose.pose.position.z
        # odom_quat = tf.quaternion_from_euler(0, 0, avg_angle)
        #header
        odom_pub_msg.header.stamp = rclpy.time.Time().to_msg()
        odom_pub_msg.header.frame_id = "/map"
        self.odom_pub.publish(odom_pub_msg)
        


def main(args=None):
    rclpy.init(args=args)
    pf = OdomNoise()
    rclpy.spin(pf)
    rclpy.shutdown()