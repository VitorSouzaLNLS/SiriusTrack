

export element_pass, line_pass, ring_pass

using ..AcceleratorModule: Accelerator
using ..Auxiliary: no_plane, plane_x, plane_xy, plane_y, pm_bnd_mpole_symplectic4_pass,
    pm_cavity_pass, pm_corrector_pass, pm_drift_pass, pm_identity_pass,
    pm_str_mpole_symplectic4_pass, st_particle_lost, st_success, vchamber_ellipse,
    vchamber_rectangle, vchamber_rhombus
using ..Elements: Element
using ..PosModule: Pos


function element_pass(
    element::Element,            # element through which to track particle
    particle::Pos{Float64},            # initial electron coordinates
    accelerator::Accelerator;    # accelerator parameters
    turn_number::Int = 0         # optional turn number parameter
    )
    status = st_success

    pass_method = element.properties[:pass_method]

    if pass_method == pm_identity_pass
        status = pm_identity_pass!(particle, element)

    elseif pass_method == pm_drift_pass
        status = pm_drift_pass!(particle, element)

    elseif pass_method == pm_str_mpole_symplectic4_pass
        status = pm_str_mpole_symplectic4_pass!(particle, element, accelerator)

    elseif pass_method == pm_bnd_mpole_symplectic4_pass
        status = pm_bnd_mpole_symplectic4_pass!(particle, element, accelerator)

    elseif pass_method == pm_corrector_pass
        status = pm_corrector_pass!(particle, element)

    elseif pass_method == pm_cavity_pass
        status = pm_cavity_pass!(particle, element, accelerator, turn_number)

    else
        return st_passmethod_not_defined
    end

    return particle, status
end

function line_pass(
    accelerator::Accelerator,
    particle::Pos{Float64},
    indices::Vector{Int};
    element_offset::Int = 1,
    turn_number::Int = 0
    )
    if any([!(1<=i<=length(accelerator.lattice)+1) for i in indices])
        leng = length(accelerator.lattice)+1
        error("invalid indices: outside of lattice bounds. The valid indices should stay between 1 and $leng")
    end
    if element_offset > length(accelerator.lattice) || element_offset < 1
        leng = length(accelerator.lattice)
        error("invalid indices: outside of lattice bounds. The valid indices should stay between 1 and $leng")
    end

    status = st_success
    lost_plane = no_plane
    tracked_pos = Pos{Float64}[]

    line = accelerator.lattice
    nr_elements = length(line)

    # Create vector of booleans to determine when to store position
    indcs = falses(nr_elements + 1)
    indcs[indices] .= true

    pos = particle

    for i in 1:nr_elements
        # Read-only access to element object parameters
        element = line[element_offset]

        # Stores trajectory at entrance of each element
        if indcs[i]
            push!(tracked_pos, copy(pos))
        end

        pos, status = element_pass(element, pos, accelerator, turn_number=turn_number)

        rx, ry = pos.rx, pos.ry

        # Checks if particle is lost
        if !isfinite(rx)
            lost_plane = plane_x
            status = st_particle_lost
        end

        if !isfinite(ry)
            if status != st_particle_lost
                lost_plane = plane_y
                status = st_particle_lost
            else
                lost_plane = plane_xy
            end
        end

        if (status != st_particle_lost) && (accelerator.vchamber_state == on)
            if element.properties[:vchamber] == vchamber_rectangle
                if rx <= element.properties[:hmin] || rx >= element.properties[:hmax]
                    lost_plane = plane_x
                    status = st_particle_lost
                end
                if ry <= element.properties[:vmin] || ry >= element.properties[:vmax]
                    if status != st_particle_lost
                        lost_plane = plane_y
                        status = st_particle_lost
                    else
                        lost_plane = plane_xy
                    end
                end
            else
                status, lost_plane = aux_check_lost_pos(element, rx, ry)
            end
        end

        if status != st_success
            # Fill the rest of vector with NaNs
            for j in i+1:Int(length(indcs))
                if indcs[j]
                    push!(tracked_pos, Pos(NaN64, NaN64, NaN64, NaN64, NaN64, NaN64))
                end
            end
            return tracked_pos, status, lost_plane
        end

        # Moves to the next element index
        element_offset = mod1(element_offset + 1, nr_elements)
    end

    # Stores final particle position at the end of the line
    if indcs[nr_elements+1]
        push!(tracked_pos, pos)
    end

    #println(stdout, "linepass posvec exit = \n$tracked_pos\n")
    return tracked_pos, status, lost_plane
end

function line_pass(
    accelerator::Accelerator,
    particle::Pos{Float64},
    indices::String = "closed";
    element_offset::Int = 1,
    turn_number::Int = 0
    )
    leng = length(accelerator.lattice)
    if indices == "closed"
        idcs = [i for i in 1:1:leng+1]
    elseif indices == "open"
        idcs = [i for i in 1:1:leng]
    elseif indices == "end"
        idcs = [leng+1]
    else
        error("invalid indices: should be a String:(closed, open, end) or a Vector{Int})")
    end
    return line_pass(accelerator, particle, idcs, element_offset=element_offset, turn_number=turn_number)
end

function ring_pass(accelerator::Accelerator, 
    particle::Pos{Float64}, 
    nr_turns::Int = 1; 
    element_offset::Int = 1,
    turn_by_turn::Bool = false
    )
    if nr_turns<1 
        error("invalid nr_turns: should be >= 1")
    end
    if turn_by_turn
        v = Pos{Float64}[]
    end
    tracked = copy(particle)
    lostplane = no_plane
    st = st_success
    for turn in 1:1:nr_turns
        tracked, st, lostplane = line_pass(accelerator, tracked, [length(accelerator.lattice)], element_offset=element_offset, turn_number=turn-1)
        if st == st_success
            if turn_by_turn
                push!(v, copy(tracked[1]))
            end
        else
            append!(v, [Pos(NaN64, NaN64, NaN64, NaN64, NaN64, NaN64) for i in 1:1:(nr_turns-turn+1)])
            break
        end
        tracked = tracked[1]
    end
    if !turn_by_turn && st == st_success
        v = [copy(tracked)]
        lostplane = no_plane
    end
    return v, st, lostplane
end

function aux_check_lost_pos(element::Element, rx::Float64, ry::Float64)
    lx = (element.properties[:hmax] - element.properties[:hmin]) / 2
    ly = (element.properties[:vmax] - element.properties[:vmin]) / 2
    xc = (element.properties[:hmax] + element.properties[:hmin]) / 2
    yc = (element.properties[:vmax] + element.properties[:vmin]) / 2
    xn = abs((rx - xc) / lx)
    yn = abs((ry - yc) / ly)
    
    if element.properties[:vchamber] == vchamber_rhombus
        amplitude = xn + yn
    elseif element.properties[:vchamber] == vchamber_ellipse
        amplitude = xn^2 + yn^2
    else
        amplitude = xn^Int(element.properties[:vchamber]) + yn^Int(element.properties[:vchamber])
    end

    if amplitude > 1
        return st_particle_lost, plane_xy
    else
        return st_success, no_plane
    end
end

