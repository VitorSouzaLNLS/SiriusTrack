# Auxiliary.jl

export string_version

macro exported_enum(name, args...)
    esc(quote
        @enum($name, $(args...))
        export $name
        $([:(export $arg) for arg in args]...)
        end)
end

@exported_enum BoolState off on

@exported_enum PassMethod pm_identity_pass pm_corrector_pass pm_drift_pass pm_matrix_pass pm_bnd_mpole_symplectic4_pass pm_str_mpole_symplectic4_pass pm_cavity_pass pm_kickmap_pass

@exported_enum VChamberShape vchamber_rectangle vchamber_rhombus vchamber_ellipse

@exported_enum RadiationState radiation_off radiation_damping radiation_full

@exported_enum Distributions dist_normal dist_uniform

@exported_enum Status st_success passmethod_not_defined passmethod_not_implemented particle_lost inconsistent_dimensions uninitialized_memory findorbit_not_converged findorbit_one_turn_matrix_problem file_not_found file_not_opened kicktable_not_defined kicktable_out_of_range flat_file_error newton_not_converged not_implemented

@exported_enum Plane no_plane plane_x plane_y plane_z plane_xy

const string_version::String = "TRACKJL version 1.0"
