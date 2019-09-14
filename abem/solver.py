from .boundary_solutions import BoundaryCondition


class Solver(object):
    def __init__(self, geometry, c=344.0, density=1.205):
        self.geometry = geometry
        self.c = c
        self.density = density

    def __repr__(self):
        result = self.__class__.__name__ + "("
        result += "  oGeometry = " + repr(self.geometry) + ", "
        result += "  c = " + repr(self.c) + ", "
        result += "  density = " + repr(self.density) + ")"
        return result

    def len(self):
        return self.geometry.len()

    def dirichlet_boundary_condition(self):
        """Returns a boundary condition with alpha the 1-function and f and beta 0-functions."""
        bc = BoundaryCondition(self.len())
        bc.alpha.fill(1.0)
        bc.beta.fill(0.0)
        bc.f.fill(1.0)
        return bc

    def neumann_boundary_condition(self):
        """Returns a boundary condition with f and alpha 0-functions and beta the 1-function."""
        bc = BoundaryCondition(self.len())
        bc.alpha.fill(0.0)
        bc.beta.fill(1.0)
        bc.f.fill(0.0)
        return bc
