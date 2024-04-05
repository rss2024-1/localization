import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs
import math
from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8

        ##### I ADDED
        self.eta = 1
        self.epsilon = 1
        self.zmax = 200

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)
        
    def hit(self, measured_val, actual_val):
        d = actual_val
        zk = measured_val
        if (0 <= zk) and (zk <= 200):
            coefficient = self.eta*(1/((2*3.14159*self.sigma_hit**2)**0.5))
            variables = math.exp(-((zk - d) ** 2) / (2 * self.sigma_hit ** 2))
            phit = coefficient * variables
        else:
            phit = 0
        return phit
    
    def short(self, measured_val, actual_val):
        d = actual_val
        zk = measured_val
        if d > 0 and (0 <= zk) and (zk <= d):
            pshort = (2/d)*(1 - zk/d)
        else:
            pshort = 0
        return pshort

    def max(self, measured_val, actual_val):
        if (measured_val >= self.zmax - self.epsilon) and (measured_val <= self.zmax):
            return 1/self.epsilon
        else:
            return 0 
        
    def rand(self, measured_val, actual_val):
        if (measured_val >= 0) and (measured_val <= self.zmax):
            return 1/self.zmax
        else:
            return 0

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        for measured_distance_val in range(self.table_width):
            for ground_truth_val in range(self.table_width):
                probability = self.alpha_hit*self.hit(measured_distance_val, ground_truth_val) + self.alpha_short*self.short(measured_distance_val, ground_truth_val) + self.alpha_max*self.max(measured_distance_val, ground_truth_val) + self.alpha_rand*self.rand(measured_distance_val, ground_truth_val)
                self.sensor_model_table[measured_distance_val][ground_truth_val] = probability

        # Assuming self.sensor_model_table is a 2D numpy array
        row_sums = self.sensor_model_table.sum(axis=1, keepdims=True)
        self.sensor_model_table = self.sensor_model_table / row_sums

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return
        
        probabilities = observation[:]
        

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles)

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
