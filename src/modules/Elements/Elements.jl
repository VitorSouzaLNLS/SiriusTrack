# Elements.jl

using ..Auxiliary: PassMethod, VChamberShape, pm_identity_pass, vchamber_rectangle

export Element

# Define Element type
mutable struct Element
    fam_name    ::String
    pass_method ::PassMethod
    length      ::Float64
    vchamber    ::VChamberShape
    polynom_a   ::Vector{Float64}
    polynom_b   ::Vector{Float64}
    nr_steps    ::Int
    angle       ::Float64
    angle_in    ::Float64
    angle_out   ::Float64
    fint_in     ::Float64
    fint_out    ::Float64
    gap         ::Float64
    hkick       ::Float64
    vkick       ::Float64
    voltage     ::Float64
    frequency   ::Float64
    phase_lag   ::Float64
    vmin        ::Float64
    vmax        ::Float64
    hmin        ::Float64
    hmax        ::Float64
    
    Element(fam_name::String) = new(
        fam_name,     # fam_name   
        pm_identity_pass,     # pass_method
        0.0,                  # length     
        vchamber_rectangle,   # vchamber   
        Vector{Float64}(),    # polynom_a  
        Vector{Float64}(),    # polynom_b  
        1,      # nr_steps   
        0.0,    # angle      
        0.0,    # angle_in   
        0.0,    # angle_out  
        0.0,    # fint_in    
        0.0,    # fint_out   
        0.0,    # gap        
        0.0,    # hkick      
        0.0,    # vkick      
        0.0,    # voltage    
        0.0,    # frequency  
        0.0,    # phase_lag  
        0.0,    # vmin
        0.0,    # vmax
        0.0,    # hmin
        0.0     # hmax  
    )
end

# Define default constructor for Element

