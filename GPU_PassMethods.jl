
using CUDA
using Adapt
using SiriusTrack.Elements: Element
using SiriusTrack.AcceleratorModule: Accelerator
using SiriusTrack.Models.StorageRing: create_accelerator
using SiriusTrack.PosModule: Pos
using SiriusTrack.Tracking: element_pass, line_pass, ring_pass
using SiriusTrack

# versioninfo()

# CUDA.versioninfo()

struct GPUElement
    fam_name   ::Int8
    pass_method::Int8
    length     ::Float64
    vchamber   ::Int8
    polynom_a  ::CuDeviceVector{Float64, 1}
    polynom_b  ::CuDeviceVector{Float64, 1}
    nr_steps   ::Int16
    angle      ::Float64
    angle_in   ::Float64
    angle_out  ::Float64
    fint_in    ::Float64
    fint_out   ::Float64
    gap        ::Float64
    hkick      ::Float64
    vkick      ::Float64
    voltage    ::Float64
    frequency  ::Float64
    phase_lag  ::Float64
    vmin       ::Float64
    vmax       ::Float64
    hmin       ::Float64
    hmax       ::Float64
end
struct GPUAccelerator 
    energy::Float64
    radiation_state::Int8
    cavity_state::Int8
    length::Float64
    velocity::Float64
    harmonic_number::Int64
    lattice::CuDeviceVector{GPUElement, 1} # still dont understant
end
Adapt.adapt_structure(to, s::Element) = GPUElement(
    adapt(to, Int8(0)),
    adapt(to, Int8(s.pass_method)),
    adapt(to, Float64(s.length)),
    adapt(to, Int8(s.vchamber)),
    adapt(to, CuArray(Float64.(s.polynom_a))),
    adapt(to, CuArray(Float64.(s.polynom_b))),
    adapt(to, Int16(s.nr_steps)),
    adapt(to, Float64(s.angle)),
    adapt(to, Float64(s.angle_in)),
    adapt(to, Float64(s.angle_out)),
    adapt(to, Float64(s.fint_in)),
    adapt(to, Float64(s.fint_out)),
    adapt(to, Float64(s.gap)),
    adapt(to, Float64(s.hkick)),
    adapt(to, Float64(s.vkick)),
    adapt(to, Float64(s.voltage)),
    adapt(to, Float64(s.frequency)),
    adapt(to, Float64(s.phase_lag)),
    adapt(to, Float64(s.vmin)),
    adapt(to, Float64(s.vmax)),
    adapt(to, Float64(s.hmin)),
    adapt(to, Float64(s.hmax))
)
Adapt.adapt_structure(to, s::Accelerator) = GPUAccelerator(
    adapt(to, Float64(s.energy)),
    adapt(to, Int8(s.radiation_state)),
    adapt(to, Int8(s.cavity_state)),
    adapt(to, Float64(s.length)),
    adapt(to, Float64(s.velocity)),
    adapt(to, Int64(s.harmonic_number)),
    adapt(to, CuArray{GPUElement, 1}([adapt(to, e) for e in s.lattice]))
)

const DRIFT1::Float64  =  0.6756035959798286638e00
const DRIFT2::Float64  = -0.1756035959798286639e00
const KICK1 ::Float64  =  0.1351207191959657328e01
const KICK2 ::Float64  = -0.1702414383919314656e01

const light_speed             ::Float64              = 299792458         # [m/s]   - definition
const electron_charge         ::Float64          = 1.602176634e-19   # [C]     - definition
const reduced_planck_constant ::Float64  = 1.054571817e-34   # [J.s]   - definition
const electron_mass           ::Float64            = 9.1093837015e-31  # [Kg]    - 2022-03-19 - https://physics.nist.gov/cgi-bin/cuu/Value?me|search_for=electron+mass
const vacuum_permeability     ::Float64      = 1.25663706212e-6  # [T.m/A] - 2022-03-19 - https://physics.nist.gov/cgi-bin/cuu/Value?mu0|search_for=vacuum+permeability
const electron_rest_energy    ::Float64     = electron_mass * light_speed^2  # [Kg.m^2/s^2] - derived
const vacuum_permitticity     ::Float64      = 1/(vacuum_permeability * light_speed^2)  # [V.s/(A.m)]  - derived
const electron_rest_energy_eV ::Float64  = (electron_rest_energy / electron_charge)  # [eV] - derived
const electron_rest_energy_MeV::Float64 = electron_rest_energy_eV / 1e6  # [MeV] - derived
const electron_rest_energy_GeV::Float64 = electron_rest_energy_eV / 1e9  # [MeV] - derived
const electron_radius         ::Float64          = electron_charge^2 / (4 * pi * vacuum_permitticity * electron_rest_energy)  # [m] - derived

const TWOPI     ::Float64 = 2*pi               # 2*pi
const CGAMMA    ::Float64 = 8.846056192e-05   # cgamma, [m]/[GeV^3] Ref[1] (4.1)
const M0C2      ::Float64 = 5.10999060e5        # Electron rest mass [eV]
const LAMBDABAR ::Float64 = 3.86159323e-13 # Compton wavelength/2pi [m]
const CER       ::Float64 = 2.81794092e-15       # Classical electron radius [m]
const CU        ::Float64 = 1.323094366892892     # 55/(24*sqrt(3)) factor
const CQEXT     ::Float64 = sqrt(CU * CER * reduced_planck_constant * electron_charge * light_speed) * electron_charge * electron_charge / ((electron_mass*light_speed*light_speed)^3) #  for quant. diff. kick

# V is a CuArray dim = (6, nr_particles)
# length is Float (actual: 32)
function _gpu_drift(V::CuDeviceArray{Float64, 2}, length::Float64)
    i::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2] # dimension of V is (6, nr_particles) -> size(V)[2] = nr_particles
        @inbounds pnorm::Float64 = 1e0 / (1e0 + V[5,i])
        norml::Float64 = length * pnorm
        @inbounds V[1,i] += (norml * V[2,i])
        @inbounds V[3,i] += (norml * V[4,i])
        @inbounds V[6,i] += (0.5e0 * norml * pnorm * (V[2,i]^2 + V[4,i]^2))
    end
    return nothing
end

# V is a CuArray dim = (6, nr_particles)
# length, rad_const and qexcit_const are Floats (32)
# polynom_a and polynom_b are CuArrays
function _gpu_strthinkick(V::CuDeviceArray{Float64, 2}, length::Float64, polynom_a::CuDeviceVector{Float64, 1}, polynom_b::CuDeviceVector{Float64, 1}, rad_const::Float64=0e0, qexit_const::Float64=0e0)
    i::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2]

        ########################## calc poly kick ##########################
        real_sum::Float64 = 0e0
        imag_sum::Float64 = 0e0
        x::Float64 = V[1,i]
        y::Float64 = V[3,i]
        n = CUDA.min(CUDA.length(polynom_a), CUDA.length(polynom_b))
        if n != 0
            @inbounds real_sum = polynom_b[n]
            @inbounds imag_sum = polynom_a[n]
            real_sum_temp::Float64 = 0e0
            for j = n-1:-1:1
                @inbounds real_sum_temp = (real_sum * x) - (imag_sum * y) + polynom_b[j]
                @inbounds imag_sum = (imag_sum * x) + (real_sum * y) + polynom_a[j]
                real_sum = real_sum_temp
            end
        end
        ####################################################################
        if rad_const != 0e0
            @inbounds de::Float64 = V[5,i]
            @inbounds pnorm::Float64 = 1e0 / (1e0 + de)
            @inbounds px::Float64 = V[2,i]*pnorm
            @inbounds py::Float64 = V[4,i]*pnorm
            ########################## calc b2p perp ##########################
            bx::Float64 = imag_sum
            by::Float64 = real_sum
            curv::Float64 = 1e0
            b2p::Float64 = (((by^2 + bx^2) * curv^2) + (bx*py - by*px)^2) / (curv^2 + px^2 + py^2)
            ###################################################################
            delta_factor::Float64 = (1e0 + de)^2
            dl_ds::Float64 = 1e0 + ((px*px + py*py) / 2e0)
            @inbounds V[5,i] -= (rad_const * delta_factor * b2p * dl_ds * length)
        
            if qexit_const != 0e0
                d::Float64 = delta_factor * qexit_const * sqrt(b2p^1.5 * dl_ds)
                @inbounds V[5,i] += d * randn(Float64)
            end

            @inbounds pnorm = 1e0 + V[5,i]
            @inbounds V[2,i] = px * pnorm
            @inbounds V[4,i] = py * pnorm
        end
        @inbounds V[2,i] -= length * real_sum
        @inbounds V[4,i] += length * imag_sum
    end
    return nothing
end

# V is a CuArray dim = (6, nr_particles)
# length, irho, rad_const and qexcit_const are Floats (32)
# polynom_a and polynom_b are CuArrays
function _gpu_bndthinkick(V::CuDeviceArray{Float64, 2}, length::Float64, polynom_a::CuDeviceVector{Float64, 1}, polynom_b::CuDeviceVector{Float64, 1}, irho::Float64, rad_const::Float64=0e0, qexit_const::Float64=0e0)
    i::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2]

        ########################## calc poly kick ##########################
        real_sum::Float64 = 0e0
        imag_sum::Float64 = 0e0
        x::Float64 = V[1,i]
        y::Float64 = V[3,i]
        n = CUDA.min(CUDA.length(polynom_a), CUDA.length(polynom_b))
        if n != 0
            @inbounds real_sum = polynom_b[n]
            @inbounds imag_sum = polynom_a[n]
            real_sum_temp::Float64 = 0e0
            for j = n-1:-1:1
                @inbounds real_sum_temp = (real_sum * x) - (imag_sum * y) + polynom_b[j]
                @inbounds imag_sum = (imag_sum * x) + (real_sum * y) + polynom_a[j]
                real_sum = real_sum_temp
            end
        end
        ####################################################################
        @inbounds de::Float64 = V[5,i]
        if rad_const != 0e0
            pnorm::Float64 = 1e0 / (1e0 + de)
            @inbounds px::Float64 = V[2,i]*pnorm
            @inbounds py::Float64 = V[4,i]*pnorm
            @inbounds curv::Float64 = 1e0 + (irho * V[1,i])
            ########################## calc b2p perp ##########################
            bx::Float64 = imag_sum
            by::Float64 = real_sum + irho
            b2p::Float64 = (((by^2 + bx^2) * curv^2) + (bx*py - by*px)^2) / (curv^2 + px^2 + py^2)
            ###################################################################
            delta_factor::Float64 = (1e0 + de)^2
            dl_ds::Float64 = curv + ((px*px + py*py) / 2)
            @inbounds V[5,i] -= (rad_const * delta_factor * b2p * dl_ds * length)
        
            if qexit_const != 0e0
                d::Float64 = delta_factor * qexit_const * sqrt(b2p^1.5 * dl_ds)
                @inbounds V[5,i] += d * randn(Float64)
            end

            @inbounds pnorm = 1e0 + V[5,i]
            @inbounds V[2,i] = px * pnorm
            @inbounds V[4,i] = py * pnorm
        end
        @inbounds V[2,i] -= length * (real_sum - (de - V[1,i] * irho) * irho)
        @inbounds V[4,i] += length * imag_sum
        @inbounds V[6,i] += length * irho * V[1,i]
    end
    return nothing
end

# V is a CuArray dim = (6, nr_particles)
# inv_rho, edge_angle, fint and gap are Floats (32)
function _gpu_edge_fringe(V::CuDeviceArray{Float64, 2}, inv_rho::Float64, edge_angle::Float64, fint::Float64, gap::Float64)
    i::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2]
        @inbounds de::Float64 = V[5,i]

        fx::Float64 = inv_rho * tan(edge_angle) / (1e0 + de)
        psi_par::Float64 = edge_angle - inv_rho * gap * fint * (1e0 + sin(edge_angle))
        fy::Float64 = inv_rho * tan(psi_par) / (1e0 + de)

        @inbounds V[2,i] += V[1,i] * fx
        @inbounds V[4,i] -= V[3,i] * fy
    end
    return nothing
end

function gpu_pm_identity_pass!(status::Int)
    status = 0 # success
    return nothing
end

function gpu_pm_drift_pass!(V::CuDeviceArray{Float64, 2}, elem, status::Int)
    _gpu_drift(V, elem.length)
    status = 0
    return nothing
end

function gpu_pm_str_mpole_symplectic4_pass!(V::CuDeviceArray{Float64, 2}, elem::GPUElement, accelerator::GPUAccelerator, status::Int)
    steps::Int16 = elem.nr_steps
    sl::Float64 = elem.length / Float64(steps)

    l1::Float64 = sl * DRIFT1
    l2::Float64 = sl * DRIFT2
    k1::Float64 = sl * KICK1
    k2::Float64 = sl * KICK2

    polynom_a::CuDeviceVector{Float64, 1} = elem.polynom_a
    polynom_b::CuDeviceVector{Float64, 1} = elem.polynom_b
    rad_const::Float64 = 0e0
    qexcit_const::Float64 = 0e0

    rad_sts::Int = Int(accelerator.radiation_state)

    if rad_sts == 1
        rad_const = CGAMMA * (accelerator.energy/1e9)^3 / TWOPI
    end
    if rad_sts == 2
        qexcit_const = CQEXT * accelerator.energy^2 * sqrt(accelerator.energy * sl)
    end

    for j in 1:steps
        _gpu_drift(V, l1)
        _gpu_strthinkick(V, k1, polynom_a, polynom_b, rad_const, 0.0e0)
        _gpu_drift(V, l2)
        _gpu_strthinkick(V, k2, polynom_a, polynom_b, rad_const, qexcit_const)
        _gpu_drift(V, l2)
        _gpu_strthinkick(V, k1, polynom_a, polynom_b, rad_const, 0.0e0)
        _gpu_drift(V, l1)
    end

    status = 0
    return nothing
end

function gpu_pm_bnd_mpole_symplectic4_pass!(V::CuDeviceArray{Float64, 2}, elem::GPUElement, accelerator::GPUAccelerator, status::Int)
    steps::Int16 = elem.nr_steps
    sl::Float64 = elem.length / steps
    l1::Float64 = sl * DRIFT1
    l2::Float64 = sl * DRIFT2
    k1::Float64 = sl * KICK1
    k2::Float64 = sl * KICK2
    irho::Float64 = elem.angle / elem.length

    polynom_a::CuDeviceVector{Float64, 1} = elem.polynom_a
    polynom_b::CuDeviceVector{Float64, 1} = elem.polynom_b
    rad_const::Float64 = 0e0
    qexcit_const::Float64 = 0e0

    rad_sts::Int = Int(accelerator.radiation_state)
    if rad_sts == 1
        rad_const = CGAMMA * (accelerator.energy/1e9)^3 / TWOPI
    end
    if rad_sts == 2
        qexcit_const = CQEXT * accelerator.energy^2 * sqrt(accelerator.energy * sl)
    end

    ang_in   ::Float64 = elem.angle_in
    ang_out  ::Float64 = elem.angle_out
    fint_in  ::Float64 = elem.fint_in
    fint_out ::Float64 = elem.fint_out
    gap      ::Float64 = elem.gap

    _gpu_edge_fringe(V, irho, ang_in, fint_in, gap)

    for _ in 1:steps
        _gpu_drift(V, l1)
        _gpu_bndthinkick(V, k1, polynom_a, polynom_b, irho, rad_const, 0.0)
        _gpu_drift(V, l2)
        _gpu_bndthinkick(V, k2, polynom_a, polynom_b, irho, rad_const, qexcit_const)
        _gpu_drift(V, l2)
        _gpu_bndthinkick(V, k1, polynom_a, polynom_b, irho, rad_const, 0.0)
        _gpu_drift(V, l1)
    end

    _gpu_edge_fringe(V, irho, ang_out, fint_out, gap)

    status = 0
    return nothing
end

function gpu_pm_corrector_pass!(V::CuDeviceArray{Float64, 2}, elem::GPUElement, status::Int)
    i::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2] # dimension of V is (6, nr_particles) -> size(V)[2] = nr_particles
        hkick::Float64 = elem.hkick
        vkick::Float64 = elem.vkick
        if elem.length == 0e0
            @inbounds V[2,i] += hkick
            @inbounds V[4,i] += vkick
        else
            px::Float64 = V[2,i]
            py::Float64 = V[4,i]
            pnorm::Float64 = 1e0 / (1e0 + V[5,i])
            norml::Float64 = elem.length * pnorm
            @inbounds V[6,i] += norml * pnorm * 0.5 * (hkick * hkick/3.0 + vkick * vkick/3.0 + px*px + py*py + px * hkick + py * vkick)
            @inbounds V[1,i] += norml * (px + 0.5 * hkick)
            @inbounds V[2,i] += hkick
            @inbounds V[3,i] += norml * (py + 0.5 * vkick)
            @inbounds V[4,i] += vkick
        end
    end
    status = 0
    return nothing
end

function gpu_pm_cavity_pass!(V::CuDeviceArray{Float64, 2}, elem::GPUElement, accelerator::GPUAccelerator, status::Int, turn_number::Int=0)
    if accelerator.cavity_state == 0
        gpu_pm_drift_pass!(V, elem, status)
        return nothing
    end
    
    i::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2] # dimension of V is (6, nr_particles) -> size(V)[2] = nr_particles

        nv::Float64 = elem.voltage / accelerator.energy
        philag::Float64 = elem.phase_lag
        frf::Float64 = elem.frequency
        harmonic_number::Int = accelerator.harmonic_number
        velocity::Float64 = accelerator.velocity
        L0::Float64 = accelerator.length
        T0::Float64 = L0 / velocity

        if elem.length == 0e0
            @inbounds V[5,i] += (-nv) * sin((TWOPI * frf * ((V[6,i]/velocity) - ((harmonic_number/frf - T0)*turn_number))) - philag)
        else
            @inbounds pnorm::Float64 = 1 / (1 + V[5,i])
            norml::Float64 = (0.5 * elem.length) * pnorm

            @inbounds V[1,i] += (norml * V[2,i])
            @inbounds V[3,i] += (norml * V[4,i])
            @inbounds V[6,i] += (0.5e0 * norml * pnorm * (V[2,i]^2 + V[4,i]^2))

            @inbounds V[5,i] += (-nv) * sin((TWOPI * frf * ((V[6,i]/velocity) - ((harmonic_number/frf - T0)*turn_number))) - philag)

            @inbounds pnorm = 1 / (1 + V[5,i])
            norml = (0.5 * elem.length) * pnorm
            @inbounds V[1,i] += (norml * V[2,i])
            @inbounds V[3,i] += (norml * V[4,i])
            @inbounds V[6,i] += (0.5e0 * norml * pnorm * (V[2,i]^2 + V[4,i]^2))
        end
        
    end
    status = 0
    return nothing
end

# PassMethods: (Int8 for GPU)
#     0 pm_identity_pass 
#     1 pm_corrector_pass 
#     2 pm_drift_pass 
#     3 pm_matrix_pass 
#     4 pm_bnd_mpole_symplectic4_pass 
#     5 pm_str_mpole_symplectic4_pass 
#     6 pm_cavity_pass 
#     7 pm_kickmap_pass

# Status: (Int for GPU)
#     0 st_success
#     2 st_passmethod_not_defined
    function gpu_element_pass_kernel(elem::GPUElement, V::CuDeviceArray{Float64, 2}, accelerator::GPUAccelerator, status::Int, turn_number::Int=0)
    pass_method::Int8 = elem.pass_method
    if pass_method == 0
        gpu_pm_identity_pass!(status)
    elseif pass_method == 2
        gpu_pm_drift_pass!(V, elem, status)
    elseif pass_method == 5
        gpu_pm_str_mpole_symplectic4_pass!(V, elem, accelerator, status)
    elseif pass_method == 4
        gpu_pm_bnd_mpole_symplectic4_pass!(V, elem, accelerator, status)
    elseif pass_method == 1
        gpu_pm_corrector_pass!(V, elem, status)
    elseif pass_method == 6
        gpu_pm_cavity_pass!(V, elem, accelerator, status, turn_number)
    else
        status = 2
    end
    return nothing
end

function gpu_line_pass_kernel(accelerator::GPUAccelerator, V::CuDeviceArray{Float64, 2}, status::Int, turn_number::Int=0)
    for elem in accelerator.lattice
        gpu_element_pass_kernel(elem, V, accelerator, status, turn_number)
    end
    return nothing
end

function gpu_ring_pass_kernel(accelerator::GPUAccelerator, V::CuDeviceArray{Float64, 2}, nr_turns::Int, status::Int)
    for n in 1:nr_turns
        gpu_line_pass_kernel(accelerator, V, status, n-1)
    end
    return nothing
end

acc = create_accelerator()
acc.radiation_state = 1
acc.cavity_state = 1
#acc.lattice = acc.lattice[1:10]
acc


p1 = Pos(0) * 1e-6
#element_pass(elem, p1, acc)
nr_turns::Int = 1024
xf, _, _ = ring_pass(acc, p1, nr_turns)

xf

function test1(saved_arr, accelerator::GPUAccelerator, V::CuDeviceArray{Float64, 2}, status::Int, turn_number::Int=0)
    i::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds saved_arr[1, i, 1] = V[1, i]
    @inbounds saved_arr[2, i, 1] = V[2, i]
    @inbounds saved_arr[3, i, 1] = V[3, i]
    @inbounds saved_arr[4, i, 1] = V[4, i]
    @inbounds saved_arr[5, i, 1] = V[5, i]
    @inbounds saved_arr[6, i, 1] = V[6, i]
    o::Int = 2
    for elem in accelerator.lattice
        gpu_element_pass_kernel(elem, V, accelerator, status, turn_number)
        @inbounds saved_arr[1, i, o] = V[1, i]
        @inbounds saved_arr[2, i, o] = V[2, i]
        @inbounds saved_arr[3, i, o] = V[3, i]
        @inbounds saved_arr[4, i, o] = V[4, i]
        @inbounds saved_arr[5, i, o] = V[5, i]
        @inbounds saved_arr[6, i, o] = V[6, i]
        o += 1
    end
    return nothing
end


const dim = Int(70*30)
nthreads = 256 #Int(floor(CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) * 0.5))
nblocks = cld(dim, nthreads)

x = CUDA.zeros(Float64, (6, dim)) * 1e-6; st::Int = 1; nturn::Int = 0; x
# saved_arr = CUDA.zeros(Float64, (6, dim, length(acc.lattice)+1))
CUDA.@time CUDA.@sync @cuda(
    threads = nthreads,
    blocks = nblocks,
    # gpu_element_pass_kernel(elem, x, acc, st, nturn)
    # gpu_pm_str_mpole_symplectic4_pass!(x, elem, acc, st)
    # gpu_line_pass_kernel(acc, x, st, nturn)
    # gpu_element_pass_kernel(elem, x, acc, st)
    #test1(saved_arr, acc, x, st, nturn)
    #test2(acc, x, st, nturn)
    gpu_ring_pass_kernel(acc, x, nr_turns, st)
); #xf[end], x[:,end]

size(x)

xf[end]
x[:, end]
