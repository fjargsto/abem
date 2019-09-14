import numpy as np


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
