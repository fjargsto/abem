import numpy as np
cimport numpy as np


cpdef float wavenumber_to_frequency(float k, float c=344.0):
    return 0.5 * k * c / np.pi


cpdef float frequency_to_wavenumber(float f, float c=344.0):
    return 2.0 * np.pi * f / c


cpdef np.complex sound_pressure(float k, np.complex  phi, float t=0.0, float c=344.0, float density=1.205):
    angularVelocity = k * c
    return (1j * density * angularVelocity * np.exp(-1.0j * angularVelocity * t)
            * phi).astype(np.complex64)


cpdef float sound_magnitude(np.complex pressure):
    return np.log10(np.abs(pressure / 2e-5)) * 20


cpdef float acoustic_intensity(np.complex pressure, np.complex velocity):
    return 0.5 * (np.conj(pressure) * velocity).real


cpdef float signal_phase(np.complex pressure):
    return np.arctan2(pressure.imag, pressure.real)
