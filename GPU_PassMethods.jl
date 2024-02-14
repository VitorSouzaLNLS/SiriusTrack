
using CUDA

const dim = 1
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


nthreads = CUDA.attribute(
    device(),
    CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
)

nblocks = cld(dim, nthreads)

# V is a CUDA Matrix, with dimensions: (6, nr_electrons)

# V is a CuArray dim = (6, nr_particles)
# length is Float (actual: 32)
function _gpu_drift(V, length)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2] # dimension of V is (6, nr_particles) -> size(V)[2] = nr_particles
        pnorm = 1f0 / (1f0 + V[5,i])
        norml = length * pnorm
        @inbounds V[1,i] += (norml * V[2,i])
        @inbounds V[3,i] += (norml * V[4,i])
        @inbounds V[6,i] += (0.5f0 * norml * pnorm * (V[2,i]^2 + V[4,i]^2))
    end
    return nothing
end

# x and y are Floats (32)
# polynom_a and polynom_b are CuArrays
function _gpu_calcpolykick(x, y, polynom_a, polynom_b)
    real_sum, imag_sum = 0f0, 0f0
    n = min(length(polynom_a), length(polynom_b))
    if n != 0
        for j in n-1:-1:1
            @inbounds real_sum_temp = real_sum * x - imag_sum * y + polynom_b[j]
            @inbounds imag_sum = imag_sum * x + real_sum * y + polynom_a[j]
            real_sum = real_sum_temp
        end
    end
    return real_sum, imag_sum
end

# bx, by, px, py and curv are Floats (32)
function _gpu_b2_perp(bx, by, px, py, curv=1f0)
    b2p = 0f0
    curv2 = curv^2
    v_norm2_inv = curv2 + px^2 + py^2
    b2p = by^2 + bx^2
    b2p *= curv2
    b2p += (bx*py - by*px)^2
    b2p /= v_norm2_inv
    return b2p
end

# V is a CuArray dim = (6, nr_particles)
# length, rad_const and qexcit_const are Floats (32)
# polynom_a and polynom_b are CuArrays
function _gpu_strthinkick(V, length, polynom_a, polynom_b, rad_const=0f0, qexit_const=0f0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2]

        real_sum, imag_sum = _gpu_calcpolykick(V[1,i], V[3,i], polynom_a, polynom_b)
        if rad_const != 0f0

            pnorm = 1f0 / (1f0 + V[5,i])
            px = V[2,i]*pnorm
            py = V[4,i]*pnorm
            b2p = _gpu_b2_perp(imag_sum, real_sum, px, py)
            delta_factor = (1f0 + V[5,i])^2
            dl_ds = 1f0 + ((px*px + py*py) / 2)
            V[5,i] -= rad_const * delta_factor * b2p * dl_ds * length
        
            if qexit_const != 0f0
                d = delta_factor * qexit_const * sqrt(b2p^1.5 * dl_ds)
                V[5,i] += d * randn(Float64)
            end

            pnorm = 1f0 + V[5,i]
            V[2,i] += px * pnorm
            V[4,i] += py * pnorm
        end
        V[2,i] -= length * real_sum
        V[4,i] += length * imag_sum
    end
    return nothing
end

# V is a CuArray dim = (6, nr_particles)
# length, irho, rad_const and qexcit_const are Floats (32)
# polynom_a and polynom_b are CuArrays
function _gpu_bndthinkick(V, length, polynom_a, polynom_b, irho, rad_const=0f0, qexit_const=0f0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2]

        real_sum, imag_sum = _gpu_calcpolykick(V[1,i], V[3,i], polynom_a, polynom_b)
        de = V[5,i]
        if rad_const != 0f0

            pnorm = 1f0 / (1f0 + de)
            px = V[2,i]*pnorm
            py = V[4,i]*pnorm
            curv = 1f0 + (irho * V[1,i])
            b2p = _gpu_b2_perp(imag_sum, real_sum, px, py, curv)
            delta_factor = (1f0 + de)^2
            dl_ds = curv + ((px*px + py*py) / 2)
            V[5,i] -= rad_const * delta_factor * b2p * dl_ds * length
        
            if qexit_const != 0f0
                d = delta_factor * qexit_const * sqrt(b2p^1.5 * dl_ds)
                V[5,i] += d * randn(Float64)
            end

            pnorm = 1f0 + V[5,i]
            V[2,i] += px * pnorm
            V[4,i] += py * pnorm
        end
        V[2,i] -= length * (real_sum - (de - V[1,i] * irho) * irho)
        V[4,i] += length * imag_sum
        V[6,i] += length * irho * V[1,i]
    end
    return nothing
end

# V is a CuArray dim = (6, nr_particles)
# inv_rho, edge_angle, fint and gap are Floats (32)
function _gpu_edge_fringe(V, inv_rho, edge_angle, fint, gap)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2]
        de = V[5,i]

        fx = inv_rho * tan(edge_angle) / (1f0 + de)
        psi_par = edge_angle - inv_rho * gap * fint * (1f0 + sin(edge_angle))
        fy = inv_rho * tan(psi_par) / (1f0 + de)

        V[2,i] += V[1,i] * fx
        V[4,i] -= V[3,i] * fy
    end
    return nothing
end
# x = CUDA.rand(Float64, (6, dim))

function gpu_pm_identity_pass!(status)
    status = 1 # success
    return nothing
end

function gpu_pm_drift_pass!(V, elem, status)
    _gpu_drift(V, elem.length)
    status = 1
    return nothing
end

function gpu_pm_str_mpole_symplectic4_pass!(V, elem, status, rad_sts, energy)
    steps = elem.nr_steps
    sl = elem.length / steps
    l1 = sl * DRIFT1
    l2 = sl * DRIFT2
    k1 = sl * KICK1
    k2 = sl * KICK2

    polynom_a = elem.polynom_a
    polynom_b = elem.polynom_b
    rad_const = 0f0
    qexcit_const = 0f0

    if rad_sts == 1
        rad_const = CGAMMA * (energy/1f9)^3 / TWOPI
    end
    if rad_sts == 2
        qexcit_const = CQEXT * energy^2 * sqrt(energy * sl)
    end

    for i in 1:steps
        _gpu_drift(V, l1)
        _gpu_strthinkick(V, k1, polynom_a, polynom_b, rad_const, 0.0f0)
        _gpu_drift(V, l2)
        _gpu_strthinkick(V, k2, polynom_a, polynom_b, rad_const, qexcit_const)
        _gpu_drift(V, l2)
        _gpu_strthinkick(V, k1, polynom_a, polynom_b, rad_const, 0.0f0)
        _gpu_drift(V, l1)
    end

    status = 1
    return nothing
end

using SiriusTrack.Elements: Element
using SiriusTrack.Models.StorageRing: create_accelerator
acc = create_accelerator()
struct GPUElement
    pass_method::Int
    length::Float64
    vchamber::Int
    polynom_a::CuDeviceVector{Float64, 1}
    polynom_b::CuDeviceVector{Float64, 1}
    nr_steps::Int
end
using Adapt
Adapt.adapt_structure(to, s::Element) = GPUElement(
    adapt(to, Int(s.pass_method)),
    adapt(to, Float64(s.length)),
    adapt(to, Int(s.vchamber)),
    adapt(to, CuArray(Float64.(s.polynom_a))),
    adapt(to, CuArray(Float64.(s.polynom_a))),
    adapt(to, Int(s.nr_steps))
)
using SiriusTrack.PosModule: Pos
using SiriusTrack.Tracking: element_pass
elem = acc.lattice[19]
p1 = Pos(1) * 1e-6
element_pass(elem, p1, acc)
p1

x = CUDA.ones(Float64, (6, dim)) * 1f-6
st = 1
CUDA.@time CUDA.@sync @cuda(
    threads = 10, #nthreads,
    blocks = 10, #nblocks,
    gpu_pm_str_mpole_symplectic4_pass!(x, elem, st, 2, 3f9)
)

p1
x