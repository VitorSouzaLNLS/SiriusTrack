# __precompile__()

module AcceleratorModule
    include("Accelerator.jl")
    using Printf

    export print_accelerator

    function print_accelerator(out::IO, accelerator::Accelerator)
        println(out, "\n--------------- Accelerator ---------------")
        println(out, "energy: $(accelerator.energy) [eV]")
        println(out, "cavity: $(accelerator.cavity_state)")
        println(out, "radiation state: $(accelerator.radiation_state)")
        println(out, "vchamber: $(accelerator.vchamber_on)")
        println(out, "harmonic number: $(accelerator.harmonic_number)")    
        println(out, "length: $(accelerator.length)")
        @printf(out, "velocity: %.8f [m/s]\n", accelerator.velocity)
        @printf(out, "beta factor: %.16f \n", accelerator.beta_factor)
        @printf(out, "gamma factor: %.1f \n", accelerator.gamma_factor)
        if (accelerator.lattice_version != "")
            println(out, "lattice version: $(accelerator.lattice_version)")
        end
        println(out, "-------------------------------------------\n")
    end

end