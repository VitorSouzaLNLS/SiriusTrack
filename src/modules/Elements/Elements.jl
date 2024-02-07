# Elements.jl

using ..Auxiliary

export Element, Element!, lattice_flatten!

# Define Element type
mutable struct Element
    fam_name::String
    properties::Dict{Symbol, Any}
end

# Define default constructor for Element
function Element!(fam_name::String)
    return Element(fam_name, Dict(:pass_method => pm_identity_pass, :length => 0.0, :vchamber => vchamber_rectangle))
end
