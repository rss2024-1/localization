

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

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
        R = np.array([[np.cos(-dtheta), -np.sin(-dtheta)],
        [np.sin(-dtheta), -np.cos(-dtheta)]])
        
        updated_particles = []
        for particle in particles: 
            new_particle = np.random.Generator.normal(loc=0.0, scale = 1.0, size=(3, )) + particle + np.append(np.dot(R, odometry[:2]), dtheta)
            updated_particles.append(new_particle)  
        
        return updated_particles

        ####################################
