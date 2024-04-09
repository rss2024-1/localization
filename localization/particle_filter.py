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

import tf2_ros 
import math

from vs_msgs.msg import ParkingError


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.particles = np.zeros((200,3))
        self.mutex = threading.Lock()

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

        self.avg_x = 0
        self.avg_y = 0
        self.avg_angle = 0

        self.noise_level = 0.3 #xy
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        
    def laser_callback(self, scan): 
        # return
        self.mutex.acquire(blocking=True)
        probabilities = self.sensor_model.evaluate(self.particles, scan.ranges)
        try: 
            probabilities /= sum(probabilities)
        except: 
            probabilities = np.empty(200)
            probabilities.fill(1/200)
        self.get_logger().info(f"probs: {probabilities}")
        resampled_indices = np.random.choice(self.particles.shape[0], size=self.particles.shape[0], replace=True, p=probabilities)
        resampled_particles = self.particles[resampled_indices]
        self.particles = resampled_particles

        self.publish_transform(self.particles)
        self.mutex.release()


    def odom_callback(self, odometry): 
        self.mutex.acquire(blocking=True)
        dt = (self.get_clock().now() - self.curr_time).nanoseconds * 1e-9
        self.curr_time = self.get_clock().now()

        dx = odometry.twist.twist.linear.x * dt
        dy = odometry.twist.twist.linear.y * dt
        dth = odometry.twist.twist.angular.z * dt

        # #add gaussian noise to velocity and angular velocity
        # dx = np.random.normal(loc=dx, scale=self.noise_level)
        # dy = np.random.normal(loc=dy, scale=self.noise_level)
        # dth = np.random.normal(loc=dth, scale=self.noise_level)

        od = [dx, dy, dth]
        self.particles = self.motion_model.evaluate(self.particles, od)

        # return updated_particles
        self.publish_transform(self.particles)
        self.mutex.release()
        
    def pose_callback(self, msg): 
        """
        var:
            msg = PoseWithCovarianceStamped Message, vars: pose, covariance
                pose = Pose Message, vars: Point position, Quaternion orientation
        """
        self.mutex.acquire(blocking=True)
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # euler fr/ q
        odom_euler = tf.euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        # self.get_logger().info(f"odom_quat is: {odom_quat}")
        th = odom_euler[2] # maybe negative of this?? idk??


        newx = np.array(x + np.random.normal(loc=0.0, scale=self.noise_level, size=(len(self.particles),1)))
        newy = np.array(y + np.random.normal(loc=0.0, scale=self.noise_level, size=(len(self.particles),1)))
        newth = np.array(th + np.random.normal(loc=0.0, scale=0.1, size=(len(self.particles),1)))
        # newx = np.full(shape=(len(self.particles),1), fill_value=x) 
        # newy = np.full(shape=(len(self.particles),1), fill_value=y)  
        # newth = np.full(shape=(len(self.particles),1), fill_value=th)  
         # np.angle(np.exp(1j * (th + np.random.default_rng().uniform(low=0.0, high=2*np.pi, size=(len(self.particles),1)))))

        # self.get_logger().info(np.array_str(newx))
        # self.get_logger().info(np.array_str(newy))
        # self.get_logger().info(np.array_str(newth))
        self.particles = np.concatenate((newx, newy, newth), axis=1)

        # self.get_logger().info("*******Initialized particles from pose******")
        self.publish_transform(self.particles)
        self.mutex.release()



    def publish_transform(self, particles): 
        # self.mutex.acquire(blocking=True)
        #average particle pose 
        sin_sum = np.sum(np.sin(particles[:,2]))
        cos_sum = np.sum(np.cos(particles[:,2]))
        avg_angle = np.arctan2(sin_sum, cos_sum)

        avg_x = np.average(particles[:,0])
        avg_y = np.average(particles[:,1])
        
        odom_pub_msg = Odometry() # creating message template in case ros is mad
        odom_pub_msg.pose.pose.position.x = avg_x 
        odom_pub_msg.pose.pose.position.y = avg_y
        odom_pub_msg.pose.pose.position.z = 0.0
        odom_quat = tf.quaternion_from_euler(0, 0, avg_angle)
        odom_pub_msg.pose.pose.orientation = Quaternion(x=odom_quat[0], y=odom_quat[1], z=odom_quat[2], w=odom_quat[3])
        #header

        odom_pub_msg.header.stamp = rclpy.time.Time().to_msg()
        odom_pub_msg.header.frame_id = "/map"
        # odom_pub_msg.header.frame_id = self.particle_filter_frame
        self.odom_pub.publish(odom_pub_msg)

        self.avg_x = avg_x
        self.avg_y = avg_y
        self.avg_angle = avg_angle


        #posearray
        particles_msg = PoseArray()
        particles_msg.header.stamp = rclpy.time.Time().to_msg()
        particles_msg.header.frame_id = "/map"
        # particles_msg.header.frame_id = self.particle_filter_frame
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
        # self.mutex.release()

        self.error_publisher()

        
    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """

        try:
            tf_map_base_link: TransformStamped = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            # mat = self.tf_to_se3(tf_map_base_link)

            odom_quat = tf.quaternion_from_euler(0, 0, self.avg_angle)
            tf_est_pose = [self.avg_x, self.avg_y, 0]
            # err = tf_map_base_link.inverseTimes(tf_est_pose)

            # self.error_pub.publish(tf_map_base_link)

            q = tf_map_base_link.transform.rotation
            q = [q.x, q.y, q.z, q.w]
            t = tf_map_base_link.transform.translation
            x, y, z = [t.x, t.y, t.z]

            t = tf_map_base_link

            msg_frame_pos = t.transform.translation
            msg_frame_quat = t.transform.rotation
            msg_frame_quat = [msg_frame_quat.x, msg_frame_quat.y,
                            msg_frame_quat.z, msg_frame_quat.w]
            msg_frame_pos = [msg_frame_pos.x, msg_frame_pos.y, msg_frame_pos.z]
            # (roll, pitch, yaw) = euler_from_quaternion(msg_frame_quat)

            error_msg = ParkingError()
            error_msg.x_error = self.avg_x - x
            error_msg.y_error = self.avg_y - y
            error_msg.distance_error = ((self.avg_x-x)**2 + (self.avg_y-y)**2)**0.5

            self.error_pub.publish(error_msg)

        except tf2_ros.TransformException:
            self.get_logger().info('no transform from /map to /base_link found')
    



    
def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
