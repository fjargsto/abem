import numpy as np
from .acoustic_properties import wavenumber_to_frequency, sound_pressure, \
    acoustic_intensity, sound_magnitude, signal_phase


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
                                                               wavenumber_to_frequency(self.k))
        res += "index   Potential               Pressure" \
               "                 Velocity                 Intensity\n\n"
        for i in range(self.phis.size):
            pressure = sound_pressure(self.k, self.phis[i], c=self.parent.c,
                                      density=self.parent.density)
            intensity = acoustic_intensity(pressure, self.velocities[i])
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
            return sound_pressure(self.k, self.phis, c=self.parent.c,
                                  density=self.parent.density)
        else:
            range = self.parent.geometry.named_partition[named_partition]
            return sound_pressure(self.k, self.phis[range[0]: range[1]],
                                  c=self.parent.c, density=self.parent.density)

    def pressure_decibel(self, named_partition=None):
        return sound_magnitude(self.pressure(named_partition))
    
    def radiation_ratio(self):
        power = 0.0
        b_power = 0.0
        for i in range(self.phis.size):
            pressure = sound_pressure(self.k, self.phis[i], c=self.parent.c,
                                      density=self.parent.density)
            power += acoustic_intensity(pressure, self.velocities[i])
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
            pressure = sound_pressure(self.boundarySolution.k, self.phis[i],
                                      c=self.boundarySolution.parent.c,
                                      density=self.boundarySolution.parent.density)
            magnitude = sound_magnitude(pressure)
            phase = signal_phase(pressure)
            result += "{:5d}  {: 1.4e}{:+1.4e}i  {: 1.4e}{:+1.4e}i  {: 1.4e} dB  {: 1.4f}\n".format( \
                i+1, self.phis[i].real, self.phis[i].imag, pressure.real, pressure.imag, magnitude, phase)
            
        return result
