# import numpy as np
# from scan_simulator_2d import PyScanSimulator2D
# # Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# # if any error re: scan_simulator_2d occurs

# from tf_transformations import euler_from_quaternion

# from nav_msgs.msg import OccupancyGrid

# import sys

# np.set_printoptions(threshold=sys.maxsize)
# import math

# class SensorModel:

#     def __init__(self, node):
#         node.declare_parameter('map_topic', "default")
#         node.declare_parameter('num_beams_per_particle', "default")
#         node.declare_parameter('scan_theta_discretization', "default")
#         node.declare_parameter('scan_field_of_view', "default")
#         node.declare_parameter('lidar_scale_to_map_scale', 1)

#         self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
#         self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
#         self.scan_theta_discretization = node.get_parameter(
#             'scan_theta_discretization').get_parameter_value().double_value
#         self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
#         self.lidar_scale_to_map_scale = node.get_parameter(
#             'lidar_scale_to_map_scale').get_parameter_value().double_value

#         ####################################
#         # Adjust these parameters
#         self.alpha_hit = 0
#         self.alpha_short = 0
#         self.alpha_max = 0
#         self.alpha_rand = 0
#         self.sigma_hit = 0
#         self.epsilon = 0

#         # Your sensor table will be a `table_width` x `table_width` np array:
#         self.table_width = 201
#         ####################################

#         node.get_logger().info("%s" % self.map_topic)
#         node.get_logger().info("%s" % self.num_beams_per_particle)
#         node.get_logger().info("%s" % self.scan_theta_discretization)
#         node.get_logger().info("%s" % self.scan_field_of_view)

#         # Precompute the sensor model table
#         self.sensor_model_table = np.empty((self.table_width, self.table_width))
#         self.precompute_sensor_model()

#         # Create a simulated laser scan
#         self.scan_sim = PyScanSimulator2D(
#             self.num_beams_per_particle,
#             self.scan_field_of_view,
#             0,  # This is not the simulator, don't add noise
#             0.01,  # This is used as an epsilon
#             self.scan_theta_discretization)

#         # Subscribe to the map
#         self.map = None
#         self.map_set = False
#         self.map_subscriber = node.create_subscription(
#             OccupancyGrid,
#             self.map_topic,
#             self.map_callback,
#             1)

#     def precompute_sensor_model(self):
#         """
#         Generate and store a table which represents the sensor model.

#         For each discrete computed range value, this provides the probability of 
#         measuring any (discrete) range. This table is indexed by the sensor model
#         at runtime by discretizing the measurements and computed ranges from
#         RangeLibc.
#         This table must be implemented as a numpy 2D array.

#         Compute the table based on class parameters alpha_hit, alpha_short,
#         alpha_max, alpha_rand, sigma_hit, and table_width.

#         args:
#             N/A

#         returns:
#             No return type. Directly modify `self.sensor_model_table`.
#         """

#         e = math.e
#         pi = math.pi
#         table = []
        
#         phit_array = np.empty((self.table_width, self.table_width))
#         pshort_array = np.empty((self.table_width, self.table_width))
#         pmax_array = np.empty((self.table_width, self.table_width))
#         prand_array = np.empty((self.table_width, self.table_width))
#         zmax =200

#         for zk_i in range(self.table_width):
#             for d in range(self.table_width):
#                 phit = (1/math.sqrt(2*pi*self.sigma_hit**2))*e**(-((zk_i - d)**2)/(2*self.sigma_hit**2)) if zk_i <= zmax else 0 
#                 pshort = (2/d)*(1 - zk_i/d) if zk_i<=d else 0 
#                 pmax = 1/self.epsilon if zk_i <= zmax and zmax - self.epsilon <= zk_i else 0 
#                 prand = 1/zmax if zk_i <= zmax else 0 

#                 phit_array[zk_i][d] = phit 
#                 pshort_array[zk_i][d] = pshort
#                 pmax_array[zk_i][d] = pmax
#                 prand_array[zk_i][d] = prand

#         for zk_i in range(self.table_width):
#             for d in range(self.table_width):
#                 p_zki = self.alpha_hit*phit + self.alpha_short*pshort + self.alpha_max*pmax + self.alpha_rand*prand
#                 table[zk_i][d] = p_zki

#         for col in range(self.table_width):
#             table[:,col] = np.normalize(table[:,col])

#         self.sensor_model_table = table


import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)

class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)
        node.declare_parameter('sensor_epsilon', 0.1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value
        self.epsilon = node.get_parameter('sensor_epsilon')

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 0.5
        self.z_max = 1
        self.n = 1
        self.epsilon = 1
        self.normalization_constant = 1

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.p_hit_table = np.empty((self.table_width, self.table_width))
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
        
        for z_i in range(self.table_width): # z_i is the rows
            for d_i in range(self.table_width): # d is the columns
                self.p_hit_table[z_i][d_i] = self.p_hit(z_i, d_i)
        self.p_hit_table /= np.linalg.norm(self.p_hit_table, axis=1, norm='max') # normalize the columns of the hit table

        for z_i in range(self.table_width):
            for d_i in range(self.table_width):
                self.sensor_model_table[z_i][d_i] = self.add_probablities(z_i, d_i, self.p_hit_table[z_i][d_i])

        self.sensor_model_table /= np.linalg.norm(self.sensor_model_table, axis=1, norm='max') # normalize the columns

    def add_probablities(self, z, d, p_hit_val):
        return self.alpha_hit*p_hit_val+self.alpha_short*self.p_short(z, d)+self.alpha_max*self.p_max(z)+self.alpha_rand*self.p_rand(z)

    def p_hit(self, z, d):
        return self.n*1/np.sqrt(2*np.pi*self.sigma_hit**2)*np.exp(-(z-d)**2/(2*self.sigma_hit**2)) if 0 <= z <= self.z_max else 0

    def p_short(self, z, d):
        return 2/d*(1-z/d) if (0<=z<=d and d!=0) else 0

    def p_max(self, z):
        return 1/self.epsilon if self.z_max-self.epsilon <= z <= self.z_max else 0

    def p_rand(self, z):
        return 1/self.z_max if 0 <= z <= self.z_max else 0        

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

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles) # an nxm array where m is num beams per particle supposedly
        
        try:
            self.num_beams_per_particle == np.shape(scans)[1]
        except AssertionError:
            print("num beams per particle doesn't match after the ray casting")

        # convert from meters to pixels
        scans = scans*self.resolution*self.lidar_scale_to_map_scale
        observation = observation*self.resolution*self.lidar_scale_to_map_scale
        
        probabilities = np.ones(len(particles))
        for i, particle in enumerate(particles):
            for j in range(self.num_beams_per_particle):
                # get index of map @ ray to lookup probability at location
                x = int(particle[0] + scans[i, j] * np.cos(particle[2] + j * self.scan_theta_discretization))
                y = int(particle[1] + scans[i, j] * np.sin(particle[2] + j * self.scan_theta_discretization))

                # check in map bounds
                if x < 0 or y < 0 or x >= self.map.shape[0] or y >= self.map.shape[1]:
                    continue

                probabilities[i] *= self.sensor_model_table[int(observation[j]), self.map[y, x]]
        
        # do we need to normalize?
        # probabilities /= np.sum(probabilities)
        return probabilities

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
