
export light_speed, electron_charge, reduced_planck_constant, electron_mass
export vacuum_permeability, electron_rest_energy, vacuum_permitticity
export electron_rest_energy_eV, electron_rest_energy_MeV, electron_rest_energy_GeV, electron_radius

const light_speed::Float64              = 299792458         # [m/s]   - definition
const electron_charge::Float64          = 1.602176634e-19   # [C]     - definition
const reduced_planck_constant::Float64  = 1.054571817e-34   # [J.s]   - definition
const electron_mass::Float64            = 9.1093837015e-31  # [Kg]    - 2022-03-19 - https://physics.nist.gov/cgi-bin/cuu/Value?me|search_for=electron+mass
const vacuum_permeability::Float64      = 1.25663706212e-6  # [T.m/A] - 2022-03-19 - https://physics.nist.gov/cgi-bin/cuu/Value?mu0|search_for=vacuum+permeability
const electron_rest_energy::Float64     = electron_mass * light_speed^2  # [Kg.m^2/s^2] - derived
const vacuum_permitticity::Float64      = 1/(vacuum_permeability * light_speed^2)  # [V.s/(A.m)]  - derived
const electron_rest_energy_eV::Float64  = (electron_rest_energy / electron_charge)  # [eV] - derived
const electron_rest_energy_MeV::Float64 = electron_rest_energy_eV / 1e6  # [MeV] - derived
const electron_rest_energy_GeV::Float64 = electron_rest_energy_eV / 1e9  # [MeV] - derived
const electron_radius::Float64          = electron_charge^2 / (4 * pi * vacuum_permitticity * electron_rest_energy)  # [m] - derived
