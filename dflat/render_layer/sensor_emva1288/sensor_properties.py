import numpy as np

ideal_sensor = {
    "name": "ideal",
    "ADC": 12,
    "QE": 1.0,
    "dark_offset": 10,
    "dark_noise_e": 0.0,
    "gain": 1,
    "well_depth_e": None,  # This is not used at the moment because we don't want to worry about well saturation
    "shot_noise": False,
    "dark_noise": False,
}


BFS_PGE_51S5 = {
    "name": "SONY IMX250, CMOS 2/3",
    "ADC": 12,
    "QE": 0.24,
    "dark_offset": 0,
    "dark_noise_e": 2.45,
    "gain": 1 / 0.18,
    "well_depth_e": None,  # This is not used at the moment because we don't want to worry about well saturation
    "shot_noise": True,
    "dark_noise": True,
}


def SNR_to_meanPhotons(SNR, sensor_emv):
    QE = sensor_emv["QE"]
    sig_dark = sensor_emv["dark_noise_e"]
    ADC = sensor_emv["ADC"]
    gain = sensor_emv["gain"]

    num_photons = SNR**2 / 2 / QE * (1 + np.sqrt(1 + 4 * (sig_dark**2 + 1 / ADC / gain**2) / SNR**2))
    return num_photons
