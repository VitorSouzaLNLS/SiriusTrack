

export track_linepass!, track_elementpass!

using ..AcceleratorModule: Accelerator
using ..Elements: Element
using ..PosModule: Pos
using ..Auxiliary: vchamber_rectangle, vchamber_rhombus, vchamber_ellipse, st_success, no_plane, plane_x, plane_y, plane_xy


function track_elementpass!(
    element::Element,            # element through which to track particle
    orig_pos::Pos{T},            # initial electron coordinates
    accelerator::Accelerator,    # accelerator parameters
    turn_number::Int = 0         # optional turn number parameter
) where T
    status = st_success

    pass_method = element.properties[:pass_method]

    if pass_method == pm_identity_pass
        status = pm_identity_pass!(orig_pos, element)

    elseif pass_method == pm_drift_pass
        status = pm_drift_pass!(orig_pos, element)

    elseif pass_method == pm_str_mpole_symplectic4_pass
        status = pm_str_mpole_symplectic4_pass!(orig_pos, element, accelerator)

    elseif pass_method == pm_bnd_mpole_symplectic4_pass
        status = pm_bnd_mpole_symplectic4_pass!(orig_pos, element, accelerator)

    elseif pass_method == pm_corrector_pass
        status = pm_corrector_pass!(orig_pos, element)

    elseif pass_method == pm_cavity_pass
        status = pm_cavity_pass!(orig_pos, element, accelerator, turn_number)

    else
        return passmethod_not_defined
    end

    return orig_pos, status
end

function track_elementpass!(
    element::Element,            # element through which to track particle
    v_pos::Vector{Pos{T}},            # initial electron coordinates
    accelerator::Accelerator,    # accelerator parameters
    turn_number::Int = 0         # optional turn number parameter
) where T
    status = st_success
    for pos in v_pos
        st = track_elementpass!(element, pos, accelerator, turn_number)
        if st != st_success
            status = st
        end
    end
    return status
end

function track_linepass!(
    accelerator::Accelerator,
    orig_pos::Pos{T},
    element_offset::Int,
    indices::Vector{Int},
    turn_number::Int = 0
) where T
    status = st_success
    lost_plane = no_plane
    tracked_pos = Pos{Float64}[]

    line = accelerator.lattice
    nr_elements = length(line)

    # Reserve space for pos
    # pos[end+1:end+length(indices)] .= Pos{T}[]

    # Create vector of booleans to determine when to store position
    indcs = falses(nr_elements + 1)
    indcs[indices] .= true

    pos = orig_pos

    for i in 1:nr_elements
        # Read-only access to element object parameters
        element = line[element_offset]

        # Stores trajectory at entrance of each element
        if indcs[i]
            push!(tracked_pos, copy(pos))
        end

        pos, status = track_elementpass!(element, pos, accelerator, turn_number)

        rx, ry = pos.rx, pos.ry

        pm = element.properties[:pass_method]
        fname = element.fam_name

        # Checks if particle is lost
        if !isfinite(rx)
            lost_plane = plane_x
            status = particle_lost
        end

        if !isfinite(ry)
            if status != particle_lost
                lost_plane = plane_y
                status = particle_lost
            else
                lost_plane = plane_xy
            end
        end

        if (status != particle_lost) && (accelerator.vchamber_on == on)
            if element.properties[:vchamber] == vchamber_rectangle
                if haskey(element.properties, :hmin) && haskey(element.properties, :hmax) haskey(element.properties, :vmin) && haskey(element.properties, :vmax)
                    if rx <= element.properties[:hmin] || rx >= element.properties[:hmax]
                        lost_plane = plane_x
                        status = particle_lost
                    end
                    if ry <= element.properties[:vmin] || ry >= element.properties[:vmax]
                        if status != particle_lost
                            lost_plane = plane_y
                            status = particle_lost
                        else
                            lost_plane = plane_xy
                        end
                    end
                elseif !haskey(element.properties, :hmin) && !haskey(element.properties, :hmax) !haskey(element.properties, :vmin) && !haskey(element.properties, :vmax)
                    status = st_success
                    lost_plane = no_plane 
                else
                    name = element.fam_name
                    throw(ErrorException("Missing vaccum chamber properties in the $name."))
                end
            else
                status, lost_plane = aux_check_lost_pos(element, rx, ry)
            end
        end

        if status != st_success
            # Fill the rest of vector with NaNs
            for j in i+1:nr_elements
                if indcs[j]
                    push!(tracked_pos, Pos(NaN, NaN, NaN, NaN, NaN, NaN))
                end
            end
            return tracked_pos, status
        end

        # Moves to the next element index
        element_offset = mod1(element_offset + 1, nr_elements)
    end

    # Stores final particle position at the end of the line
    if indcs[nr_elements]
        push!(tracked_pos, pos)
    end

    return tracked_pos, status
end

function aux_check_lost_pos(element::Element, rx::T, ry::T) where T
    if haskey(element.properties, :hmin) && haskey(element.properties, :hmax) haskey(element.properties, :vmin) && haskey(element.properties, :vmax) 
        lx = (element.properties[:hmax] - element.properties[:hmin]) / 2
        ly = (element.properties[:vmax] - element.properties[:vmin]) / 2
        xc = (element.properties[:hmax] + element.properties[:hmin]) / 2
        yc = (element.properties[:vmax] + element.properties[:vmin]) / 2
        xn = abs((rx - xc) / lx)
        yn = abs((ry - yc) / ly)
        
        amplitude = if element.properties[:vchamber] == vchamber_rhombus
            xn + yn
        elseif element.properties[:vchamber] == vchamber_ellipse
            xn^2 + yn^2
        else
            xn^element.properties[:vchamber] + yn^element.properties[:vchamber]
        end

        if amplitude > 1
            return particle_lost, plane_xy
        else
            return st_success, no_plane
        end
    elseif !haskey(element.properties, :hmin) && !haskey(element.properties, :hmax) !haskey(element.properties, :vmin) && !haskey(element.properties, :vmax)
        return st_success, no_plane 
    else
        name = element.fam_name
        throw(ErrorException("Missing vaccum chamber properties in the $name."))
    end
end

