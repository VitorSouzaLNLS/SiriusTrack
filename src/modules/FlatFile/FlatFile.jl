export read_flatfile!, write_flatfile!

using ..Auxiliary
using ..Elements: Element!, Element
using ..AcceleratorModule: Accelerator, Accelerator!, print_accelerator
using ..Base:TTY

function write_flatfile!() end


"""bbb."""
function read_flatfile!(file_path::AbstractString)
    # Open the file
    file = open(file_path, "r")
    
    # Initialize variables to store Accelerator properties
    energy = 0.0
    harmonic_number = 0
    cavity_on = false
    radiation_on = false
    vchamber_on = false
    lattice_version = ""
    lattice = Element[]
    current_element = nothing

    # Read each line in the file
    for line in eachline(file)
        # Check if it's a comment or section header
        if startswith(line, "%")
            # Extract properties from comments
            if contains(line, "energy")
                energy = parse(Float64, split(line)[end-1])
            elseif contains(line, "harmonic_number")
                harmonic_number = parse(Int, split(line)[end])
            elseif contains(line, "cavity_on")
                cavity_on = parse(Bool, split(line)[end])
            elseif contains(line, "radiation_on")
                radiation_on = parse(Int, split(line)[end])
            elseif contains(line, "vchamber_on")
                vchamber_on = parse(Bool, split(line)[end])
            elseif contains(line, "lattice_version")
                lattice_version = strip(split(line)[end], ['+'])           
            end
        elseif startswith(line, "###")
            # Finish processing the previous element and start a new one
            if current_element !== nothing
                push!(lattice, current_element)
            end
            current_element = nothing
        elseif isempty(line)
            # Blank line indicates the end of an element
            if current_element !== nothing
                if !haskey(current_element.properties, :polynom_b) && (current_element.properties[:pass_method] == Auxiliary.pm_bnd_mpole_symplectic4_pass || current_element.properties[:pass_method] == Auxiliary.pm_str_mpole_symplectic4_pass)
                    current_element.properties[:pass_method] = Auxiliary.pm_drift_pass
                end
                if (current_element.properties[:pass_method] == Auxiliary.pm_str_mpole_symplectic4_pass || current_element.properties[:pass_method] == Auxiliary.pm_bnd_mpole_symplectic4_pass)
                    if !haskey(current_element.properties, :polynom_b)
                        current_element.properties[:polynom_b] = Float64[0.0, 0.0, 0.0]
                    end
                    if !haskey(current_element.properties, :polynom_a)
                        current_element.properties[:polynom_a] = zeros(Float64, length(current_element.properties[:polynom_b]))
                    end
                end
                push!(lattice, current_element)
            end
            current_element = nothing
        else
            # read the properties
            if contains(line, "fam_name")
                fam_name = String(strip(split(line)[end]))
                current_element = Element!(fam_name)
            elseif contains(line, "length")
                value = parse(Float64, split(line)[end])
                current_element.properties[:length] = value
            elseif contains(line, "hmin")
                value = parse(Float64, split(line)[end])
                current_element.properties[:hmin] = value
            elseif contains(line, "hmax")
                value = parse(Float64, split(line)[end])
                current_element.properties[:hmax] = value
            elseif contains(line, "vmin")
                value = parse(Float64, split(line)[end])
                current_element.properties[:vmin] = value
            elseif contains(line, "vmax")
                value = parse(Float64, split(line)[end])
                current_element.properties[:vmax] = value
            elseif contains(line, "nr_steps")
                value = parse(Int, split(line)[end])
                current_element.properties[:nr_steps] = value
            elseif contains(line, "gap")
                value = parse(Float64, split(line)[end])
                current_element.properties[:gap] = value
            elseif contains(line, "angle")
                value = parse(Float64, split(line)[end])
                current_element.properties[:angle] = value
            elseif contains(line, "angle_in")
                value = parse(Float64, split(line)[end])
                current_element.properties[:angle_in] = value
            elseif contains(line, "angle_out")
                value = parse(Float64, split(line)[end])
                current_element.properties[:angle_out] = value
            elseif contains(line, "fint_in")
                value = parse(Float64, split(line)[end])
                current_element.properties[:fint_in] = value
            elseif contains(line, "fint_out")
                value = parse(Float64, split(line)[end])
                current_element.properties[:fint_out] = value
            elseif contains(line, "vchamber")
                value = parse(Int, split(line)[end])
                value = parse_vchamber(Int(value))
                current_element.properties[:vchamber] = value
            elseif contains(line, "voltage")
                value = parse(Float64, split(line)[end])
                current_element.properties[:voltage] = value
            elseif contains(line, "frequency")
                value = parse(Float64, split(line)[end])
                current_element.properties[:frequency] = value
            elseif startswith(line, "pass_method")
                # Parse and map pass_method to the corresponding enum
                pass_method_str = String(strip(split(line)[end]))
                pass_method = parse_pass_method(pass_method_str)
                current_element.properties[:pass_method] = pass_method
            elseif startswith(line, "polynom_b")
                polynom_b = parse_polynom(line)
                current_element.properties[:polynom_b] = polynom_b
                # if !haskey(current_element.properties, :polynom_a)
                #     current_element.properties[:polynom_a] = zeros(Float64, length(polynom_b))
                # end
            elseif startswith(line, "polynom_a")
                polynom_a = parse_polynom(line)
                current_element.properties[:polynom_a] = polynom_a
                # if !haskey(current_element.properties, :polynom_b)
                #     current_element.properties[:polynom_b] = zeros(Float64, length(polynom_a))
                # end
            end
        end
    end
    
    # Close the file
    close(file)

    
    # Create and return the Accelerator object
    acc = Accelerator!(Float64(energy))
    acc.cavity_state = Auxiliary.BoolState(Int(cavity_on))
    acc.radiation_state = Auxiliary.RadiationState(Int(radiation_on))
    acc.vchamber_on = Auxiliary.BoolState(Int(vchamber_on))
    setfield!(acc, :lattice_version, String(lattice_version))
    acc.harmonic_number = Int(harmonic_number)
    acc.lattice = lattice
    return acc
end

function parse_pass_method(pass_method_str::String)
    pass_method_mapping = Dict(
        "identity_pass" => Auxiliary.pm_identity_pass,
        "corrector_pass" => Auxiliary.pm_corrector_pass,
        "drift_pass" => Auxiliary.pm_drift_pass,
        "matrix_pass" => Auxiliary.pm_matrix_pass,
        "bnd_mpole_symplectic4_pass" => Auxiliary.pm_bnd_mpole_symplectic4_pass,
        "str_mpole_symplectic4_pass" => Auxiliary.pm_str_mpole_symplectic4_pass,
        "cavity_pass" => Auxiliary.pm_cavity_pass,
        "kickmap_pass" => Auxiliary.pm_kickmap_pass )
    pass_method_str = String(pass_method_str)
    if haskey(pass_method_mapping, pass_method_str)
        return pass_method_mapping[pass_method_str]
    end
    return Auxiliary.pm_identity_pass
end

function parse_vchamber(vchamber_int::Int)
    vchamber_mapping = Dict(
        0 => Auxiliary.vchamber_rectangle,
        1 => Auxiliary.vchamber_rhombus,
        2 => Auxiliary.vchamber_ellipse
    )
    if haskey(vchamber_mapping, vchamber_int)
        return vchamber_mapping[vchamber_int]
    end
    return Auxiliary.vchamber_rectangle
end

function parse_polynom(line::AbstractString)
    parts = split(line)
    # Extract order and value pairs starting from index 2
    order_value_pairs = parts[2:end]
    
    # Initialize the vector with zeros
    max_order = parse(Int, order_value_pairs[1:2:end][end])
    polynom_values = zeros(Float64, max_order+1)
    
    # Fill in the values based on order and value pairs
    for i in 1:2:length(order_value_pairs)
        order = parse(Int, order_value_pairs[i])
        value = parse(Float64, order_value_pairs[i + 1])
        polynom_values[order+1] = value
    end
    
    return polynom_values
end