import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.noise_level = 0.3

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO
        dx, dy, dtheta = odometry
        
        cos_thetas = np.cos(particles[:, 2])
        sin_thetas = np.sin(particles[:, 2])

        dx_rotated = cos_thetas * dx - sin_thetas * dy
        dy_rotated = sin_thetas * dx + cos_thetas * dy

        particles[:, 0] += dx_rotated + np.random.normal(loc=0.0, scale = self.noise_level, size=(len(particles),))
        particles[:, 1] += dy_rotated + np.random.normal(loc=0.0, scale = self.noise_level, size=(len(particles),))
        particles[:, 2] += dtheta + np.random.normal(loc=0.0, scale=0.01, size=(len(particles),))

        # Normalize angles to the range [-pi, pi]
        particles[:, 2] = (particles[:, 2] + np.pi) % (2 * np.pi) - np.pi        

        return particles

        ####################################

