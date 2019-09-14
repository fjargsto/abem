import numpy as np


def wavenumber_to_frequency(k, c=344.0):
    return 0.5 * k * c / np.pi


def frequency_to_wavenumber(f, c=344.0):
    return 2.0 * np.pi * f / c


def sound_pressure(k, phi, t=0.0, c=344.0, density=1.205):
    angularVelocity = k * c
    return (1j * density * angularVelocity * np.exp(-1.0j * angularVelocity * t)
            * phi).astype(np.complex64)


def sound_magnitude(pressure):
    return np.log10(np.abs(pressure / 2e-5)) * 20


def acoustic_intensity(pressure, velocity):
    return 0.5 * (np.conj(pressure) * velocity).real


def signal_phase(pressure):
    return np.arctan2(pressure.imag, pressure.real)
