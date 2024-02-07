module StorageRing

    const default_optics_mode::String = "S05.01"

    include("model.jl")

    export create_accelerator!

    function create_accelerator!(;optics_mode::String=default_optics_mode, simplified::Bool=false, ids=[])
        return create_lattice(optics_mode, simplified, ids)
    end
end