# __precompile__()

module SiriusTrack

include("modules/Pos/posModule.jl")

include("modules/Constants/constantsModule.jl")

include("modules/Auxiliary/auxiliaryModule.jl")

include("modules/Elements/elementsModule.jl")

include("modules/Accelerator/acceleratorModule.jl")

include("modules/Tracking/trackingModule.jl")

include("modules/FlatFile/flatfileModule.jl")

include("modules/Models/modelsModule.jl")

include("modules/Orbit/orbitModule.jl")

using .PosModule
using .Constants
using .Auxiliary
using .Elements
using .AcceleratorModule
using .Tracking
using .FlatFile
using .Models
using .Orbit

using PrecompileTools

@setup_workload begin
    flatfile_path::String=(String(@__DIR__))*"/modules/FlatFile/example_flatfile.txt"
    @compile_workload begin
        p = PosModule.pos(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        m = Models.StorageRing.create_accelerator()
        m.radiation_state = Auxiliary.full
        m.cavity_state = Auxiliary.on
        m.vchamber_state = Auxiliary.on
        pf, st, lf = Tracking.ring_pass(m, p, 1)
        mff = FlatFile.read_flatfile(flatfile_path)
        Orbit.find_orbit4(m)
        Orbit.find_orbit6(m)
    end
end

end # module SiriusTrack
