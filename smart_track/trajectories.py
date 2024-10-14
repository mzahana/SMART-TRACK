import numpy as np

class Circle3D:
    def __init__(self, normal_vector, center_vector, radius=1, omega=1):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.center_vector = center_vector
        self.radius = radius
        self.omega = omega

        not_parallel = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else np.array([0, 1, 0])
        self.v1 = np.cross(self.normal_vector, not_parallel)
        self.v1 /= np.linalg.norm(self.v1)
        self.v2 = np.cross(self.normal_vector, self.v1)
        self.v2 /= np.linalg.norm(self.v2)

    def generate_trajectory_setpoint(self, time):
        t = self.omega * time
        point = self.center_vector + self.radius * (np.cos(t) * self.v1 + np.sin(t) * self.v2)
        return point
    
    def updateParameters(self,  normal_vector, center_vector, radius=1, omega=1):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.center_vector = center_vector
        self.radius = radius
        self.omega = omega

        not_parallel = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else np.array([0, 1, 0])
        self.v1 = np.cross(self.normal_vector, not_parallel)
        self.v1 /= np.linalg.norm(self.v1)
        self.v2 = np.cross(self.normal_vector, self.v1)
        self.v2 /= np.linalg.norm(self.v2)

    def timeToCompleteFullTrajectory(self):
        return 3 * np.pi / self.omega

# circle = Circle3D(np.array([1, 2, 3]), np.array([2, 3, 4]), radius=2, omega=1)

class Infinity3D:
    def __init__(self, normal_vector, center_vector, radius=1, omega=1):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.center_vector = center_vector
        self.radius = radius
        self.omega = omega

        not_parallel = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else np.array([0, 1, 0])
        self.v1 = np.cross(self.normal_vector, not_parallel)
        self.v1 /= np.linalg.norm(self.v1)
        self.v2 = np.cross(self.normal_vector, self.v1)
        self.v2 /= np.linalg.norm(self.v2)

    def generate_trajectory_setpoint(self, time):
        t = self.omega * time
        point = self.center_vector + self.radius * (np.cos(t) * self.v1 + np.sin(2*t) * self.v2)
        return point
    
    def updateParameters(self,  normal_vector, center_vector, radius=1, omega=1):
        self.normal_vector = normal_vector / np.linalg.norm(normal_vector)
        self.center_vector = center_vector
        self.radius = radius
        self.omega = omega

        not_parallel = np.array([1, 0, 0]) if self.normal_vector[0] == 0 else np.array([0, 1, 0])
        self.v1 = np.cross(self.normal_vector, not_parallel)
        self.v1 /= np.linalg.norm(self.v1)
        self.v2 = np.cross(self.normal_vector, self.v1)
        self.v2 /= np.linalg.norm(self.v2)

    def timeToCompleteFullTrajectory(self):
        return 4 * np.pi / self.omega

# infinity = Infinity3D(np.array([1, 2, 3]), np.array([2, 3, 4]), radius=2, omega=1)