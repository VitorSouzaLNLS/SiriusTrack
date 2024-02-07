# Accelerator.jl

import Base: !=, setproperty!, setfield!, getproperty
using ..Auxiliary
using ..Constants
using ..Elements: Element

export Accelerator, Accelerator!, find_spos, find_indices

electron_rest_energy_eV = Constants.electron_rest_energy_eV

mutable struct Accelerator
    energy::Float64
    cavity_state::Auxiliary.BoolState
    radiation_state::Auxiliary.RadiationState
    vchamber_on::Auxiliary.BoolState
    harmonic_number::Int
    lattice::Vector{Element}
    lattice_version::String
    length::Float64
    velocity::Float64
    beta_factor::Float64
    gamma_factor::Float64
end    

function Accelerator!(energy::Real)
    gamma = energy/electron_rest_energy_eV
    beta = sqrt(1 - (1 / gamma^2))
    velocity = beta * light_speed
    accelerator = Accelerator(energy, Auxiliary.off, Auxiliary.radiation_off, Auxiliary.off, 0, Element[], "", 0.0, velocity, beta, gamma)
    return accelerator
end

function isequal(acc1::Accelerator, acc2::Accelerator)
    if (acc1.energy != acc2.energy) return false end
    if (acc1.cavity_state != acc2.cavity_state) return false end
    if (acc1.radiation_state != acc2.radiation_state) return false end
    if (acc1.vchamber_on != acc2.vchamber_on) return false end
    if (acc1.harmonic_number != acc2.harmonic_number) return false end
    if (acc1.lattice_version != acc2.lattice_version) return false end
    #if (acc1.lattice != acc2.lattice) return false end revisar comparador de lattices
    return true
end

function Base.:(!=)(acc1::Accelerator, acc2::Accelerator)
    return !(isequal(acc1, acc2))
end

function update_cavity(accelerator::Accelerator)
    cavity_indices = find_cavity(accelerator)
    for index in cavity_indices
        cav = accelerator.lattice[index]
        if accelerator.cavity_state == on
            cav.properties[:pass_method] = Auxiliary.pm_cavity_pass
        elseif accelerator.cavity_state == off && cav.length == 0.0
            cav.properties[:pass_method] = Auxiliary.pm_identity_pass
        else
            cav.properties[:pass_method] = Auxiliary.pm_drift_pass
        end
    end
end

function setproperty!(accelerator::Accelerator, symbol::Symbol, value)
    if symbol == :energy
        # Custom logic for setting the energy field
        if !(value <= electron_rest_energy_eV)
            #accelerator.energy = value
            setfield!(accelerator, :energy, value)

            gamma = value/electron_rest_energy_eV
            setfield!(accelerator, :gamma_factor, gamma)

            beta = sqrt(1 - (1 / gamma^2))
            setfield!(accelerator, :beta_factor, beta)

            velocity = beta * light_speed
            setfield!(accelerator, :velocity, velocity)
        end
    
    elseif symbol == :cavity_state
        # Custom logic for setting the cavity_state field
        if value == true
            setfield!(accelerator, :cavity_state, on)
        elseif value == false
            setfield!(accelerator, :cavity_state, off)
        elseif value == Auxiliary.on || value == Auxiliary.off
            setfield!(accelerator, :cavity_state, value)
        else
            throw(ArgumentError("Invalid argument for $symbol"))
        update_cavity(accelerator)
        end
    
    elseif symbol == :radiation_state
        # Custom logic for setting the radiation_on field
        if isa(value, Auxiliary.RadiationState) || isa(value, Int) || isa(value, Bool)
            setfield!(accelerator, :radiation_state, Auxiliary.RadiationState(Int(value)))
        end
    
    elseif symbol == :vchamber_on
        # Custom logic for setting the vchamber_on field
        setfield!(accelerator, :vchamber_on, value)
    
    elseif symbol == :harmonic_number
        # Custom logic for setting the harmonic_number field
        setfield!(accelerator, :harmonic_number, value)

    elseif symbol == :lattice
        # Custom logic for setting the lattice field
        setfield!(accelerator, :lattice, value)
        last_index = Int(length(value))+1
        len = find_spos(accelerator, last_index)[1] # closed lattice
        setfield!(accelerator, :length, len)

    elseif symbol == :lattice_version
        # Custom logic for setting the lattice_version field
        # setfield!(accelerator, :lattice_version, value)
        @warn("Changing the \"lattice_version\" manually is not recommended..")
    
    elseif symbol == :length # Do nothing -> automatic calculation
        # Custom logic for setting the length field
        # accelerator.length = value
        @warn("Cant manually change the \"length\". Consider changing the accelerator's lattice.")
        # setfield!(accelerator, :length, value)
    
    elseif symbol == :velocity # Do nothing -> automatic calculation
        # Custom logic for setting the velocity field
        #accelerator.velocity = value
        # @warn("Cant manually change the \"velocity\". Consider changing the accelerator's energy.")
        # #update_lorentz_factors(accelerator)

    elseif symbol == :beta_factor # Do nothing -> automatic calculation
        # Custom logic for setting the beta_factor field
        #accelerator.beta_factor = value
        # @warn("Cant manually change the \"beta_factor\". Consider changing the accelerator's energy.")
        # #update_lorentz_factors(accelerator)

    elseif symbol == :gamma_factor # Do nothing -> automatic calculation
        # Custom logic for setting the gamma_factor field
        #accelerator.gamma_factor = value
        # @warn("Cant manually change the \"gamma_factor\". Consider changing the accelerator's energy.")
        #update_lorentz_factors(accelerator)
    else
        throw(ArgumentError("Field $symbol is not a valid field for Accelerator"))
    end
end

function _all_spos(accelerator::Accelerator)
    spos = Float64[]
    temp_spos = 0.0
    latt = accelerator.lattice
    for elem in latt
        push!(spos, temp_spos)
        temp_spos += elem.properties[:length]
    end
    push!(spos, temp_spos)
    return spos
end

function find_spos(accelerator::Accelerator, indices::Vector{Int})
    spos = Float64[]
    all_spos = _all_spos(accelerator)
    for index in indices
        push!(spos, all_spos[index])
    end
    return spos
end

function find_spos(accelerator::Accelerator, index::Int)
    if 1 <= index <= length(accelerator.lattice)+1
        return find_spos(accelerator, [index])
    else
        throw(ArgumentError("Invalid index $index for the lattice"))
    end
end

function find_spos(accelerator::Accelerator; indices::T="open") where T<:Union{String, Vector{Int}}
    idx = Vector{Int}(range(1, length(accelerator.lattice)))
    if indices == "open"
        return find_spos(accelerator, idx)
    elseif indices == "closed"
        push!(idx, Int(length(accelerator.lattice)+1))
        return find_spos(accelerator, idx)
    else
        return find_spos(accelerator, indices)
    end
end

function find_indices(accelerator::Accelerator, property::String, value::Union{Real, String, Auxiliary.PassMethod})
    indices = Int[]
    for (idx, element) in enumerate(accelerator.lattice)
        if property == "fam_name" && element.fam_name == value
            push!(indices, idx)
        elseif haskey(element.properties, Symbol(property)) && element.properties[Symbol(property)] == value
            push!(indices, idx)
        end
    end
    return indices
end

function lattice_shift!(accelerator::Accelerator, index::Int)
    lattice::Vector{Element} = accelerator.lattice
    
    # Check if the index is within bounds
    if index < 1 || index > length(lattice)
        throw(ArgumentError("Index out of bounds"))
    end
    
    # Shift the lattice
    new_lattice = vcat(lattice[index:end], lattice[1:index-1])

    accelerator.lattice = new_lattice
end

# precompile(find_indices, (String, Union{Real, String, Auxiliary.PassMethod}, ))
# precompile(Accelerator!, (Real, ))