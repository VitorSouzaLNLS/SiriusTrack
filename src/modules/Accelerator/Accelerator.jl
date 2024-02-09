# Accelerator.jl

import Base: !=, setproperty!, setfield!, getproperty, show
using Printf
using ..Auxiliary
using ..Constants
using ..Elements: Element

#export Accelerator, Accelerator!, find_spos, find_indices

electron_rest_energy_eV = Constants.electron_rest_energy_eV

mutable struct Accelerator
    energy::Float64
    cavity_state::Auxiliary.BoolState
    radiation_state::Auxiliary.BoolState
    vchamber_state::Auxiliary.BoolState
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
    accelerator = Accelerator(energy, Auxiliary.off, Auxiliary.off, Auxiliary.off, 0, Element[], "", 0.0, velocity, beta, gamma)
    return accelerator
end

function isequal(acc1::Accelerator, acc2::Accelerator)
    if (acc1.energy != acc2.energy) return false end
    if (acc1.cavity_state != acc2.cavity_state) return false end
    if (acc1.radiation_state != acc2.radiation_state) return false end
    if (acc1.vchamber_state != acc2.vchamber_state) return false end
    if (acc1.harmonic_number != acc2.harmonic_number) return false end
    if (acc1.lattice_version != acc2.lattice_version) return false end
    #if (acc1.lattice != acc2.lattice) return false end revisar comparador de lattices
    return true
end

function Base.:(!=)(acc1::Accelerator, acc2::Accelerator)
    return !(isequal(acc1, acc2))
end

function update_cavity(accelerator::Accelerator)
    cavity_indices = find_cav_indices(accelerator)
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
        if isa(value, Auxiliary.BoolState) || isa(value, Int) || isa(value, Bool)
            val = Int(value)
            if 0 <= val <= 1
                setfield!(accelerator, :cavity_state, Auxiliary.BoolState(val))
                update_cavity(accelerator)
            else
                error("cavity_state should be 0(cavity off) or 1(cavity on)")
            end
        end
    
    elseif symbol == :radiation_state
        # Custom logic for setting the radiation_on field
        if isa(value, Auxiliary.BoolState) || isa(value, Int) || isa(value, Bool)
            val = Int(value)
            if 0 <= val <= 2
                setfield!(accelerator, :radiation_state, Auxiliary.BoolState(val))
            else
                error("radiation_state should be 0(radiation off), 1(radiation dumping) or 2(radiation full)")
            end
        end
    
    elseif symbol == :vchamber_state
        # Custom logic for setting the vchamber_state field
        setfield!(accelerator, :vchamber_state, value)
    
    elseif symbol == :harmonic_number
        # Custom logic for setting the harmonic_number field
        setfield!(accelerator, :harmonic_number, value)

    elseif symbol == :lattice
        # Custom logic for setting the lattice field
        setfield!(accelerator, :lattice, value)
        last_index = Int(length(value))+1
        len = find_spos(accelerator, indices="closed")[end] # closed lattice
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

function find_spos(accelerator::Accelerator; indices::T="open") where T<:Union{String, Vector{Int}}
    spos::Vector{Float64} = append!([0.0], [elem.properties[:length] for elem in accelerator.lattice])
    spos = cumsum(spos)
    if isa(indices, String)
        if indices == "open"
            return spos[1:end-1]
        elseif indices == "closed"
            return spos
        else
            throw(ArgumentError("invalid indices: should be (String)\"closed\" or \"open\" or (Vector{Int})"))
        end
    elseif isa(indices, Vector{Int})
        if all([(1<=i<=accelerator.length) for i in indices])
            return spos[indices]
        else
            leng = length(accelerator.lattice)
            throw(ArgumentError("invalid index in argument \"indices\": should stay between 1 and $leng"))
        end
    else
        throw(ArgumentError("invalid indices: should be (String)\"closed\" or \"open\" or (Vector{Int})"))
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

function find_cav_indices(accelerator::Accelerator)
    idcs = Int[]
    for i in 1:1:length(accelerator.lattice) 
        if haskey(accelerator.lattice[i].properties, :frequency)
            push!(idcs, i)
        end
    end
    return idcs
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

function Base.show(io::IO, ::MIME"text/plain", accelerator::Accelerator)
    println(io, "---------------- Accelerator ---------------")
    println(io, "\tEnergy: ", accelerator.energy)
    println(io, "\tCavity State: ", accelerator.cavity_state)
    println(io, "\tRadiation State: ", accelerator.radiation_state)
    println(io, "\tVchamber State: ", accelerator.vchamber_state)
    println(io, "\tHarmonic Number: ", accelerator.harmonic_number)
    println(io, "\tLength: ", accelerator.length)
    @printf(io, "\tVelocity: %.8f [m/s]\n", accelerator.velocity)
    @printf(io, "\tBeta Factor: %.16f \n", accelerator.beta_factor)
    @printf(io, "\tGamma Factor: %.1f \n", accelerator.gamma_factor)
    if (accelerator.lattice_version != "")
        println(io, "\tLattice Version: $(accelerator.lattice_version)")
    else
        println(io, "Lattice Version: (none)")
    end
    println(io, "--------------------------------------------")
end