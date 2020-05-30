import numpy as np
import acoustic_properties

# -----------------------------------------------------------------------------
# Boundary representations
# -----------------------------------------------------------------------------
class Chain(object):
    def __init__(self, num_vertices=0, num_edges=0):
        self.vertices = np.empty((num_vertices, 2), dtype=np.float32)
        self.edges = np.empty((num_edges, 2), dtype=np.int32)
        # named partitions are duples of start and end indices into the triangle array
        self.named_partition = {}
        self._centers = None
        self._lengths = None
        self._areas = None
        self._normals = None

    def __repr__(self):
        result = self.__class__.__name__ + "(\n"
        result += "aVertex({}) = {},\n ".format(self.vertices.shape[0], self.vertices)
        result += "aEdge({}) = {}, \n".format(self.edges.shape[0], self.edges)
        result += "namedPartition = {}\n)".format(self.named_partition)
        return result

    def edge_vertices(self, edge):
        return self.vertices[self.edges[edge, 0]], \
               self.vertices[self.edges[edge, 1]]

    def len(self):
        return self.edges.shape[0]

    def centers(self):
        if self._centers is None:
            self._centers = (self.vertices[self.edges[:, 0]] +
                             self.vertices[self.edges[:, 1]]) / 2.0
        return self._centers

    def _compute_lengths_and_normals(self):
        # length of the boundary elements
        self._lengths = np.empty(self.edges.shape[0], dtype=np.float32)
        self._normals = np.empty((self.edges.shape[0], 2), dtype=np.float32)
        for i in range(self._lengths.size):
            a = self.vertices[self.edges[i, 0], :]
            b = self.vertices[self.edges[i, 1], :]
            ab = b - a
            normal = np.empty_like(ab)
            normal[0] = ab[1]
            normal[1] = -ab[0]
            length = np.linalg.norm(normal)
            self._normals[i] = normal / length
            self._lengths[i] = length

    def lengths(self):
        if self._lengths is None:
            self._compute_lengths_and_normals()
        return self._lengths

    def normals(self):
        if self._normals is None:
            self._compute_lengths_and_normals()
        return self._normals

    def areas(self, named_partition = None):
        """The areas of the surfaces created by rotating an edge around the x-axis."""
        if self._areas is None:
            self._areas = np.empty(self.edges.shape[0], dtype=np.float32)
            for i in range(self._areas.size):
                a, b = self.edge_vertices(i)
                self._areas[i] = np.pi * (a[0] + b[0]) * np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        if named_partition is None:
            return self._areas
        else:
            partition = self.named_partition[named_partition]
            return self._areas[partition[0]: partition[1]]

    
class Mesh(object):
    def __init__(self, num_vertices=0, num_triangles=0):
        self.vertices = np.empty((num_vertices, 3), dtype=np.float32)
        self.triangles = np.empty((num_triangles, 3), dtype=np.int32)
        # named partitions are duples of start and end indices into the triangle array
        self.named_partition = {}
        self._centers = None
        self._areas = None
        self._normals = None

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "aVertex = {}, ".format(self.vertices)
        result += "aTriangle = {}, ".format(self.triangles)
        result += "namedPartition = {})".format(self.named_partition)
        return result

    def triangle_vertices(self, triangle):
        return self.vertices[self.triangles[triangle, 0]], \
               self.vertices[self.triangles[triangle, 1]], \
               self.vertices[self.triangles[triangle, 2]]

    def len(self):
        return self.triangles.shape[0]

    def centers(self):
        if self._centers is None:
            self._centers = (self.vertices[self.triangles[:, 0]] +
                             self.vertices[self.triangles[:, 1]] +
                             self.vertices[self.triangles[:, 2]]) / 3.0
        return self._centers

    def _compute_areas_and_normals(self):
        # area of the boundary elements
        self._areas = np.empty(self.len(), dtype=np.float32)
        self._normals = np.empty((self.len(), 3), dtype=np.float32)
        for i in range(self._areas.size):
            a = self.vertices[self.triangles[i, 0], :]
            b = self.vertices[self.triangles[i, 1], :]
            c = self.vertices[self.triangles[i, 2], :]
            ab = b - a
            ac = c - a
            normal = np.cross(ab, ac)
            length = np.linalg.norm(normal)
            self._normals[i] = normal / length
            self._areas[i] = 0.5 * length

    def areas(self, named_partition=None):
        assert named_partition is None, "Named partitions not yet supported."
        if self._areas is None:
            self._compute_areas_and_normals()
        return self._areas

    def normals(self):
        if self._normals is None:
            self._compute_areas_and_normals()
        return self._normals


# -----------------------------------------------------------------------------
# Boundary factories
# -----------------------------------------------------------------------------
def square():
    chain = Chain(32, 32)
    chain.vertices = np.array([[0.00, 0.0000], [0.00, 0.0125], [0.00, 0.0250], [0.00, 0.0375],
                               [0.00, 0.0500], [0.00, 0.0625], [0.00, 0.0750], [0.00, 0.0875],

                               [0.0000, 0.10], [0.0125, 0.10], [0.0250, 0.10], [0.0375, 0.10],
                               [0.0500, 0.10], [0.0625, 0.10], [0.0750, 0.10], [0.0875, 0.10],

                               [0.10, 0.1000], [0.10, 0.0875], [0.10, 0.0750], [0.10, 0.0625],
                               [0.10, 0.0500], [0.10, 0.0375], [0.10, 0.0250], [0.10, 0.0125],

                               [0.1000, 0.00], [0.0875, 0.00], [0.0750, 0.00], [0.0625, 0.00],
                               [0.0500, 0.00], [0.0375, 0.00], [0.0250, 0.00], [0.0125, 0.00]],
                              dtype=np.float32)

    chain.edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                            [4, 5], [5, 6], [6, 7], [7, 8],

                            [8,  9],  [9,  10], [10, 11], [11, 12],
                            [12, 13], [13, 14], [14, 15], [15, 16],

                            [16, 17], [17, 18], [18, 19], [19, 20],
                            [20, 21], [21, 22], [22, 23], [23, 24],

                            [24, 25], [25, 26], [26, 27], [27, 28],
                            [28, 29], [29, 30], [30, 31], [31,  0]],
                           dtype=np.int32)
    return chain


# For RIM3T1 (Rayleigh Integral test program 1)
# This is the 2D "piston" or circular membrane lying in the
# plane {(x,y,z)|x = 1.0, y,z in R}
#
def disk_3d():
    mesh = Mesh(19, 25)
    mesh.vertices = np.array([[1.000, 0.000, 0.000],   [1.000, 0.025, 0.043],
                              [1.000, 0.050, 0.000],   [1.000,  0.025, -0.043],
                              [1.000, -0.025, -0.043], [1.000, -0.050, 0.000],
                              [1.000, -0.025,  0.043], [1.000,  0.050,  0.087],
                              [1.000,  0.087, 0.050],  [1.000,  0.100,  0.000],
                              [1.000,  0.087, -0.050], [1.000,  0.050, -0.087],
                              [1.000,  0.000, -0.100], [1.000, -0.050, -0.087],
                              [1.000, -0.087, -0.050], [1.000, -0.100,  0.000],
                              [1.000, -0.087,  0.050], [1.000, -0.050, 0.087],
                              [1.000,  0.000,  0.100]], dtype=np.float32)

    mesh.triangles = np.array([[0, 1, 6],   [0, 2, 1],   [0, 3, 2],   [0, 4, 3],
                               [0, 5,  4],  [0, 6,  5],  [7, 1,  8],  [8, 1,  2],
                               [8, 2,  9],  [9, 2, 10],  [10, 2,  3], [10, 3, 11],
                               [11, 3, 12], [12, 3,  4], [12, 4, 13], [13, 4, 14],
                               [14, 4,  5], [14, 5, 15], [15, 5, 16], [16, 5, 6],
                               [16, 6, 17], [17, 6, 18], [18, 7, 1],  [18, 1, 7]])
    return mesh


def square_3d():
    mesh = Mesh(25, 32)
    mesh.vertices = np.array([[0.000, 0.000, 0.000], [0.250, 0.000, 0.000],
                              [0.500, 0.000, 0.000], [0.750, 0.000, 0.000],
                              [1.000, 0.000, 0.000], [0.000, 0.250, 0.000],
                              [0.250, 0.250, 0.000], [0.500, 0.250, 0.000],
                              [0.750, 0.250, 0.000], [1.000, 0.250, 0.000],
                              [0.000, 0.500, 0.000], [0.250, 0.500, 0.000],
                              [0.500, 0.500, 0.000], [0.750, 0.500, 0.000],
                              [1.000, 0.500, 0.000], [0.000, 0.750, 0.000],
                              [0.250, 0.750, 0.000], [0.500, 0.750, 0.000],
                              [0.750, 0.750, 0.000], [1.000, 0.750, 0.000],
                              [0.000, 1.000, 0.000], [0.250, 1.000, 0.000],
                              [0.500, 1.000, 0.000], [0.750, 1.000, 0.000],
                              [1.000, 1.000, 0.000]])

    mesh.triangles = np.array([[0,   1,  5], [1,   6,  5], [1,   2,  7], [1,   7,  6],
                               [2,   3,  7], [3,   8,  7], [3,   4,  9], [3,   9,  8],
                               [5,   6, 11], [5,  11, 10], [6,   7, 11], [7,  12, 11],
                               [7,   8, 13], [7,  13, 12], [8,   9, 13], [9,  14, 13],
                               [10, 11, 15], [11, 16, 15], [11, 12, 17], [11, 17, 16],
                               [12, 13, 17], [13, 18, 17], [13, 14, 19], [13, 19, 18],
                               [15, 16, 21], [15, 21, 20], [16, 17, 21], [17, 22, 21],
                               [17, 18, 23], [17, 23, 22], [18, 19, 23], [19, 24, 23]])
    return mesh


def sphere_rad():
    oChain = Chain(19, 18)
    oChain.vertices = np.array([[0.000,  1.000], [0.174,  0.985], [0.342,  0.940], [0.500,  0.866],
                                [0.643,  0.766], [0.766,  0.643], [0.866,  0.500], [0.940,  0.342],
                                [0.985,  0.174], [1.000,  0.000], [0.985, -0.174], [0.940, -0.342],
                                [0.866, -0.500], [0.766, -0.643], [0.643, -0.766], [0.500, -0.866],
                                [0.342, -0.940], [0.174, -0.985], [0.000, -1.000]], dtype=np.float32)

    oChain.edges = np.array([[0,   1], [1,   2], [2,   3], [3,   4],
                             [4,   5], [5,   6], [6,   7], [7,   8],
                             [8,   9], [9,  10], [10, 11], [11, 12],
                             [12, 13], [13, 14], [14, 15], [15, 16],
                             [16, 17], [17, 18]], dtype=np.int32)
    return oChain


def sphere():
    mesh = Mesh(20, 36)
    mesh.vertices = np.array([[0.000,   0.000,   1.000],
                              [0.000,   0.745,   0.667],
                              [0.645,   0.372,   0.667],
                              [0.645,  -0.372,   0.667],
                              [0.000,  -0.745,   0.667],
                              [-0.645,  -0.372,  0.667],
                              [-0.645,   0.372,  0.667],
                              [0.500,    0.866,  0.000],
                              [1.000,    0.000,  0.000],
                              [0.500,   -0.866,  0.000],
                              [-0.500,  -0.866,  0.000],
                              [-1.000,   0.000,  0.000],
                              [-0.500,   0.866,  0.000],
                              [0.000,    0.745, -0.667],
                              [0.645,    0.372, -0.667],
                              [0.645,   -0.372, -0.667],
                              [0.000,   -0.745, -0.667],
                              [-0.645,  -0.372, -0.667],
                              [-0.645,   0.372, -0.667],
                              [0.000,    0.000, -1.000]], dtype=np.float32)

    mesh.triangles = np.array([[0,   2,  1], [0,   3,  2], [0,   4,  3],
                               [0,   5,  4], [0,   6,  5], [0,   1,  6],
                               [1,   2,  7], [2,   8,  7], [2,   3,  8],
                               [3,   9,  8], [3,   4,  9], [4,  10,  9],
                               [4,   5, 10], [5,  11, 10], [5,   6, 11],
                               [6,  12, 11], [6,   1, 12], [1,   7, 12],
                               [7,  14, 13], [7,   8, 14], [8,  15, 14],
                               [8,   9, 15], [9,  16, 15], [9,  10, 16],
                               [10, 17, 16], [10, 11, 17], [11, 18, 17],
                               [11, 12, 18], [12, 13, 18], [12,  7, 13],
                               [13, 14, 19], [14, 15, 19], [15, 16, 19],
                               [16, 17, 19], [17, 18, 19], [18, 13, 19]], dtype=np.int32)
    return mesh


def truncated_sphere():
    mesh = Mesh(20, 36)
    mesh.vertices = np.array([[0.000,   0.000,  0.667],
                              [0.000,   0.745,  0.667],
                              [0.645,   0.372,  0.667],
                              [0.645,  -0.372,  0.667],
                              [0.000,  -0.745,  0.667],
                              [-0.645, -0.372,  0.667],
                              [-0.645,  0.372,  0.667],
                              [0.500,   0.866,  0.000],
                              [1.000,   0.000,  0.000],
                              [0.500,  -0.866,  0.000],
                              [-0.500, -0.866,  0.000],
                              [-1.000,  0.000,  0.000],
                              [-0.500,  0.866,  0.000],
                              [0.000,   0.745, -0.667],
                              [0.645,   0.372, -0.667],
                              [0.645,  -0.372, -0.667],
                              [0.000,  -0.745, -0.667],
                              [-0.645, -0.372, -0.667],
                              [-0.645,  0.372, -0.667],
                              [0.000,   0.000, -1.000]], dtype=np.float32)

    mesh.triangles = np.array([[0,   2,  1], [0,   3,  2], [0,   4,  3],
                               [0,   5,  4], [0,   6,  5], [0,   1,  6],
                               [1,   2,  7], [2,   8,  7], [2,   3,  8],
                               [3,   9,  8], [3,   4,  9], [4,  10,  9],
                               [4,   5, 10], [5,  11, 10], [5,   6, 11],
                               [6,  12, 11], [6,   1, 12], [1,   7, 12],
                               [7,  14, 13], [7,   8, 14], [8,  15, 14],
                               [8,   9, 15], [9,  16, 15], [9,  10, 16],
                               [10, 17, 16], [10, 11, 17], [11, 18, 17],
                               [11, 12, 18], [12, 13, 18], [12,  7, 13],
                               [13, 14, 19], [14, 15, 19], [15, 16, 19],
                               [16, 17, 19], [17, 18, 19], [18, 13, 19]], dtype=np.int32)
    open_triangles = 6
    mesh.named_partition['interface'] = (0, open_triangles)
    mesh.named_partition['cavity'] = (open_triangles, 36)
    return mesh

def truncated_sphere_rad():
    chain = Chain(15, 14)
    chain.vertices = np.array([[0.000,   0.000],
                                [0.200,  0.000],
                                [0.400,  0.000],
                                [0.600,  0.000],
                                [0.800,  0.000],
                                [1.000,  0.000],
                                [0.985, -0.174],
                                [0.940, -0.342],
                                [0.866, -0.500],
                                [0.766, -0.643],
                                [0.643, -0.766],
                                [0.500, -0.866],
                                [0.342, -0.940],
                                [0.174, -0.985],
                                [0.000, -1.000]], dtype=np.float32)

    chain.edges = np.array([[0,   1],
                            [1,   2],
                            [2,   3],
                            [3,   4],
                            [4,   5],
                            [5,   6],
                            [6,   7],
                            [7,   8],
                            [8,   9],
                            [9,  10],
                            [10, 11],
                            [11, 12],
                            [12, 13],
                            [13,  0]], dtype=np.int32)
    open_edges = 5
    chain.named_partition['interface'] = (0, open_edges)
    chain.named_partition['cavity'] = (open_edges, 14)

    return chain


# -----------------------------------------------------------------------------
# Boundary solutions
# -----------------------------------------------------------------------------

class BoundaryCondition(object):
    def __init__(self, size):
        self.alpha = np.empty(size, dtype=np.complex64)
        self.beta = np.empty(size, dtype=np.complex64)
        self.f = np.empty(size, dtype=np.complex64)

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "alpha = {}, ".format(self.alpha)
        result += "beta = {}, ".format(self.beta)
        result += "f = {})".format(self.f)
        return result


class BoundaryIncidence(object):
    def __init__(self, size):
        self.phi = np.empty(size, dtype=np.complex64)
        self.v = np.empty(size, dtype=np.complex64)


class BoundarySolution(object):
    def __init__(self, parent, boundary_condition, k, phis, velocities):
        self.parent = parent
        self.boundary_condition = boundary_condition
        self.k = k
        self.phis = phis
        self.velocities = velocities

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "parent = " + repr(self.parent) + ", "
        result += "boundary_condition = " + repr(self.boundary_condition) + ", "
        result += "k = " + repr(self.k) + ", "
        result += "aPhi = " + repr(self.phis) + ", "
        result += "aV = " + repr(self.velocities) + ")"
        return result

    def __str__(self):
        res = "Density of medium:      {} kg/m^3\n".format(self.parent.density)
        res += "Speed of sound:         {} m/s\n".format(self.parent.c)
        res += "Wavenumber (Frequency): {} ({} Hz)\n\n".format(self.k,
                                                               acoustic_properties.wavenumber_to_frequency(self.k))
        res += "index   Potential               Pressure" \
               "                 Velocity                 Intensity\n\n"
        for i in range(self.phis.size):
            pressure = acoustic_properties.sound_pressure(self.k, self.phis[i], c=self.parent.c,
                                      density=self.parent.density)
            intensity = acoustic_properties.acoustic_intensity(pressure, self.velocities[i])
            res += "{:5d}  {: 1.4e}{:+1.4e}  {: 1.4e}{:+1.4e}i  " \
                   "{: 1.4e}{:+1.4e}i  {: 1.4e}\n".format(i+1,
                                                          self.phis[i].real, self.phis[i].imag,
                                                          pressure.real, pressure.imag,
                                                          self.velocities[i].real,
                                                          self.velocities[i].imag,
                                                          intensity)
        return res

    def pressure(self, named_partition=None):
        if named_partition is None:
            return acoustic_properties.sound_pressure(self.k, self.phis, c=self.parent.c,
                                  density=self.parent.density)
        else:
            range = self.parent.geometry.named_partition[named_partition]
            return acoustic_properties.sound_pressure(self.k, self.phis[range[0]: range[1]],
                                  c=self.parent.c, density=self.parent.density)

    def pressure_decibel(self, named_partition=None):
        return acoustic_properties.sound_magnitude(self.pressure(named_partition))

    def radiation_ratio(self):
        power = 0.0
        b_power = 0.0
        for i in range(self.phis.size):
            pressure = acoustic_properties.sound_pressure(self.k, self.phis[i], c=self.parent.c,
                                      density=self.parent.density)
            power += acoustic_properties.acoustic_intensity(pressure, self.velocities[i])
            b_power += (self.parent.density * self.parent.c *
                        np.abs(self.velocities[i]) ** 2)
        return 2.0 * power / b_power

    def mechanical_impedance(self, named_partition=None):
        Zm = 0.0
        if named_partition is None:
            aV = self.velocities
        else:
            partition = self.parent.geometry.named_partition[named_partition]
            aV = self.velocities[partition[0]:partition[1]]
        for p, a, v in zip(self.pressure(named_partition),
                           self.parent.element_area(named_partition),
                           aV):
            Zm += p * a / v
        return Zm


class InteriorBoundarySolution(BoundarySolution):
    def solve_samples(self, incident_phis, points):
        return SampleSolution(self,
                              self.parent.solve_samples(self, incident_phis,
                                                        points, 'interior'))


class ExteriorBoundarySolution(BoundarySolution):
    def solve_samples(self, incident_phis, points):
        return SampleSolution(self,
                              self.parent.solve_samples(self, incident_phis,
                                                        points, 'exterior'))


class RayleighBoundarySolution(BoundarySolution):
    def solve_samples(self, points):
        return self.parent.solve_samples(self, points)


class RayleighCavityBoundarySolution(BoundarySolution):
    def solve_cavity(self, points):
        """Solve for point internal to the cavity."""
        return self.parent.solve_interior(self, points)

    def solve_samples(self, points):
        """Solve for points in half space (exterior)."""
        return self.parent.solve_exterior(self, points)


class SampleSolution(object):
    def __init__(self, boundary_solution, phis):
        self.boundarySolution = boundary_solution
        self.phis = phis

    def __repr__(self):
        result = "SampleSolution("
        result += "boundarySolution = " + repr(self.parent) + ", "
        result += "aPhi = " + repr(self.phis) + ")"
        return result

    def __str__(self):
        result = "index   Potential                Pressure" \
                 "                 Magnitude       Phase\n\n"
        for i in range(self.phis.size):
            pressure = acoustic_properties.sound_pressure(self.boundarySolution.k, self.phis[i],
                                      c=self.boundarySolution.parent.c,
                                      density=self.boundarySolution.parent.density)
            magnitude = acoustic_properties.sound_magnitude(pressure)
            phase = acoustic_properties.signal_phase(pressure)
            result += "{:5d}  {: 1.4e}{:+1.4e}i  {: 1.4e}{:+1.4e}i  {: 1.4e} dB  {: 1.4f}\n".format( \
                i+1, self.phis[i].real, self.phis[i].imag, pressure.real, pressure.imag, magnitude, phase)

        return result
