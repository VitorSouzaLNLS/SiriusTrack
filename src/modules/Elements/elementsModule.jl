# __precompile__()

module Elements

include("Elements.jl")

export corrector, marker, hcorrector, vcorrector, drift, matrix, rbend, quadrupole, sextupole, rfcavity, kickmap, lattice_flatten!, print_element

# Define functions for Element
function marker(fam_name::String)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_identity_pass
    return element
end

function corrector(fam_name::String, hkick::Float64, vkick::Float64)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_corrector_pass
    element.properties[:hkick] = hkick
    element.properties[:vkick] = vkick
    return element
end

function hcorrector(fam_name::String, hkick::Float64)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_corrector_pass
    element.properties[:hkick] = hkick
    element.properties[:vkick] = 0.0
    return element
end

function vcorrector(fam_name::String, vkick::Float64)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_corrector_pass
    element.properties[:hkick] = 0.0
    element.properties[:vkick] = vkick
    return element
end

function drift(fam_name::String, length::Float64)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_drift_pass
    element.properties[:length] = length
    return element
end

function matrix(fam_name::String, length::Float64)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_matrix_pass
    element.properties[:length] = length
    return element
end

function rbend(fam_name::String, length::Float64, angle::Float64, angle_in::Float64, angle_out::Float64,
                gap::Float64, fint_in::Float64, fint_out::Float64, polynom_a::Vector{Float64},
                polynom_b::Vector{Float64}, K::Float64=-999.0, S::Float64=-999.0, nr_steps::Int=20)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_bnd_mpole_symplectic4_pass
    element.properties[:length] = length
    element.properties[:angle] = angle
    element.properties[:angle_in] = angle_in
    element.properties[:angle_out] = angle_out
    element.properties[:gap] = gap
    element.properties[:fint_in] = fint_in
    element.properties[:fint_out] = fint_out
    element.properties[:polynom_a] = polynom_a
    element.properties[:polynom_b] = polynom_b
    if (K != -999.0) 
        element.properties[:polynom_b][2] = K
    end
    if (S != -999.0) 
        element.properties[:polynom_b][3] = S
    end
    element.properties[:nr_steps] = nr_steps
    return element
end

function quadrupole(fam_name::String, length::Float64, K::Float64; nr_steps::Int=10)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_str_mpole_symplectic4_pass
    element.properties[:polynom_a] = [0.0, 0.0, 0.0] # copy(default_polynom)
    element.properties[:polynom_b] = [0.0, 0.0, 0.0] # copy(default_polynom) 
    element.properties[:polynom_b][2] = K
    element.properties[:nr_steps] = nr_steps
    element.properties[:length] = length
    return element
end

function sextupole(fam_name::String, length::Float64, S::Float64; nr_steps::Int=5)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_str_mpole_symplectic4_pass
    element.properties[:polynom_a] = [0.0, 0.0, 0.0] # copy(default_polynom)
    element.properties[:polynom_b] = [0.0, 0.0, 0.0] # copy(default_polynom) 
    element.properties[:polynom_b][3] = S
    element.properties[:nr_steps] = nr_steps
    element.properties[:length] = length
    return element
end

function rfcavity(fam_name::String, frequency::Float64, voltage::Float64, phase_lag::Float64)  
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_cavity_pass
    element.properties[:frequency] = frequency
    element.properties[:voltage] = voltage
    element.properties[:phase_lag] = phase_lag
    return element
end

# Revisar implementacao
function kickmap(fam_name::String, kicktable_idx::Int, nr_steps::Int, rescale_kicks::Float64)
    element = Element!(fam_name)
    element.properties[:pass_method] = pm_kickmap_pass
    element.properties[:nr_steps] = nr_steps
    element.properties[:kicktable_idx] = kicktable_idx
    element.properties[:rescale_kicks] = rescale_kicks
    return element
end

function flatten_to_element_vector(arg::Any)
    if isa(arg, Vector{Element})
        return arg
    elseif isa(arg, Element)
        return [arg]
    elseif isa(arg, Vector)
        # Recursively flatten each element in the vector
        flattened = []
        for elem in arg
            append!(flattened, flatten_to_element_vector(elem))
        end
        return flattened
    else
        throw(ArgumentError("Unsupported argument type: $(typeof(arg))"))
    end
end

function lattice_flatten!(arg::Any)
    latt = Vector{Element}(flatten_to_element_vector(arg))
    return latt
end

# Other functions
function print_element(out::IO, element::Element)
    custom_order = [:pass_method, :nr_steps, :angle, :angle_in, :angle_out, :gap, :fint_in, :fint_out, :polynom_a, :polynom_b, :frequency, :voltage, :phase_lag, :kicktable_idx, :rescale_kicks, :hkick, :vkick]
    
    keys_sorted = [key in keys(element.properties) ? key : nothing for key in custom_order]
    keys_sorted = filter(x -> x !== nothing, keys_sorted)

    name = element.fam_name
    println(out, "$name")
    for key in keys_sorted
        value = element.properties[key]
        println(out, "$key : $value")
    end
end

end #