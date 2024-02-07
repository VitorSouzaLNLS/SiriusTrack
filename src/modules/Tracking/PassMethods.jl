
export pm_drift_pass!, pm_str_mpole_symplectic4_pass!, pm_bnd_mpole_symplectic4_pass!, pm_corrector_pass!, pm_cavity_pass!

import Base: !=, getproperty, setfield!, setproperty!

using ..AcceleratorModule: Accelerator
using ..Auxiliary: no_plane, off, on, particle_lost, plane_x, plane_xy, plane_y, plane_z,
    pm_bnd_mpole_symplectic4_pass, pm_cavity_pass, pm_corrector_pass, pm_drift_pass,
    pm_identity_pass, pm_str_mpole_symplectic4_pass, radiation_damping, radiation_full,
    radiation_off, st_success
using ..Elements: Element
using ..PosModule: Pos
using ..Constants: light_speed

const DRIFT1::Float64  =  0.6756035959798286638e00
const DRIFT2::Float64  = -0.1756035959798286639e00
const KICK1::Float64   =  0.1351207191959657328e01
const KICK2::Float64   = -0.1702414383919314656e01

const TWOPI::Float64 = 2*pi               # 2*pi
const CGAMMA::Float64 = 8.846056192e-05   # cgamma, [m]/[GeV^3] Ref[1] (4.1)
const M0C2::Float64 = 5.10999060e5        # Electron rest mass [eV]
const LAMBDABAR::Float64 = 3.86159323e-13 # Compton wavelength/2pi [m]
const CER::Float64 = 2.81794092e-15       # Classical electron radius [m]
const CU::Float64 = 1.323094366892892     # 55/(24*sqrt(3)) factor

function aux_drift(pos::Pos{T}, length::T) where T
    pnorm::Float64 = 1 / (1 + pos.de)
    norml::Float64 = length * pnorm
    pos.rx += norml * pos.px
    pos.ry += norml * pos.py
    pos.dl += 0.5 * norml * pnorm * (pos.px*pos.px + pos.py*pos.py)
end

# function aux_drift_fast(pos::Pos{T}, norml::T) where T
#     pos.rx += norml * pos.px
#     pos.ry += norml * pos.py
#     pos.dl += 0.5 * norml * (pos.px^2 + pos.py^2) / (1 + pos.de)
# end

function aux_calcpolykick(pos::Pos{T}, polynom_a::Vector{Float64},
    polynom_b::Vector{Float64}) where T
    n = min(length(polynom_b), length(polynom_a))
    if n == 0
        real_sum = 0.0
        imag_sum = 0.0
    else
        real_sum = polynom_b[n]
        imag_sum = polynom_a[n]
        for i = n-1:-1:1
            real_sum_tmp = real_sum * pos.rx - imag_sum * pos.ry + polynom_b[i]
            imag_sum = imag_sum * pos.rx + real_sum * pos.ry + polynom_a[i]
            real_sum = real_sum_tmp
        end
    end
    return real_sum, imag_sum
end

function aux_b2_perp(bx::T, by::T, px::T, py::T, curv::T=1.0) where T
    curv2 = curv^2
    v_norm2_inv = curv2 + px^2 + py^2
    b2p = by^2 + bx^2
    b2p *= curv2
    b2p += (bx * py - by * px)^2
    b2p /= v_norm2_inv
    return b2p
end

function aux_strthinkick(pos::Pos{T}, length::Float64, polynom_a::Vector{Float64},
    polynom_b::Vector{Float64}, rad_const::Float64=0.0, qexcit_const::Float64=0.0) where T

    real_sum, imag_sum = aux_calcpolykick(pos, polynom_a, polynom_b)

    if rad_const != 0.0
        pnorm = 1 / (1 + pos.de)
        rx, px, ry, py = pos.rx, pos.px * pnorm, pos.ry, pos.py * pnorm
        b2p = aux_b2_perp(imag_sum, real_sum, px, py)
        delta_factor = (1 + pos.de)^2
        dl_ds = 1.0 + ((px*px + py*py) / 2)
        pos.de -= rad_const * delta_factor * b2p * dl_ds * length

        if qexcit_const != 0.0
            # quantum excitation kick
            d = delta_factor * qexcit_const * sqrt(b2p^1.5 * dl_ds)
            pos.de += d * randn()
        end

        pnorm = 1.0 + pos.de  # actually, this is the inverse of pnorm
        pos.px = px * pnorm
        pos.py = py * pnorm
    end

    pos.px -= length * real_sum
    pos.py += length * imag_sum
end

function aux_bndthinkick(pos::Pos{T}, length::Float64, polynom_a::Vector{Float64},
    polynom_b::Vector{Float64}, irho::Float64, rad_const::Float64=0.0, qexcit_const::Float64=0.0) where T
    
    real_sum, imag_sum = aux_calcpolykick(pos, polynom_a, polynom_b)
    de = copy(pos.de)

    if rad_const != 0
        pnorm = 1 / (1 + pos.de)
        rx, px, ry, py = pos.rx, pos.px * pnorm, pos.ry, pos.py * pnorm
        curv = 1.0 + (irho * rx)
        b2p = aux_b2_perp(imag_sum, real_sum + irho, px, py, curv)
        delta_factor = (1 + pos.de)^2
        dl_ds = curv + ((px*px + py*py) / 2)
        pos.de -= rad_const * delta_factor * b2p * dl_ds * length

        if qexcit_const != 0
            # quantum excitation kick
            d = delta_factor * qexcit_const * sqrt(b2p^1.5 * dl_ds)
            pos.de += d * randn()
        end

        pnorm = 1.0 + pos.de  # actually this is the inverse of pnorm
        pos.px = px * pnorm
        pos.py = py * pnorm
    end
    
    pos.px -= length * (real_sum - (de - pos.rx * irho) * irho)
    pos.py += length * imag_sum
    pos.dl += length * irho * pos.rx
end

function aux_edge_fringe(pos::Pos{T}, inv_rho::Float64, edge_angle::Float64,
    fint::Float64, gap::Float64) where T
    rx::Float64 = pos.rx
    ry::Float64 = pos.px
    de::Float64 = pos.de

    fx = inv_rho * tan(edge_angle) / (1.0 + de)

    psi_bar = edge_angle - inv_rho * gap * fint * (1 + sin(edge_angle)^2) / cos(edge_angle) / (1.0 + de)
    
    fy = inv_rho * tan(psi_bar) / (1.0 + de)
    
    pos.px += rx * fx
    pos.py -= ry * fy
end

# function aux_translate_pos(pos::Pos{T}, t::Vector{Float64}) where T
#     pos.rx += t[1]
#     pos.px += t[2]
#     pos.ry += t[3]
#     pos.py += t[4]
#     pos.de += t[5]
#     pos.dl += t[6]
# end

# function aux_rotate_pos(pos::Pos{T}, R::Vector{Float64}) where T
#     rx0, px0 = pos.rx, pos.px
#     ry0, py0 = pos.ry, pos.py
#     de0, dl0 = pos.de, pos.dl

#     pos.rx = R[0*6 + 1]*rx0 + R[0*6 + 2]*px0 + R[0*6 + 3]*ry0 + R[0*6 + 4]*py0 + R[0*6 + 5]*de0 + R[0*6 + 6]*dl0
#     pos.px = R[1*6 + 1]*rx0 + R[1*6 + 2]*px0 + R[1*6 + 3]*ry0 + R[1*6 + 4]*py0 + R[1*6 + 5]*de0 + R[1*6 + 6]*dl0
#     pos.ry = R[2*6 + 1]*rx0 + R[2*6 + 2]*px0 + R[2*6 + 3]*ry0 + R[2*6 + 4]*py0 + R[2*6 + 5]*de0 + R[2*6 + 6]*dl0
#     pos.py = R[3*6 + 1]*rx0 + R[3*6 + 2]*px0 + R[3*6 + 3]*ry0 + R[3*6 + 4]*py0 + R[3*6 + 5]*de0 + R[3*6 + 6]*dl0
#     pos.de = R[4*6 + 1]*rx0 + R[4*6 + 2]*px0 + R[4*6 + 3]*ry0 + R[4*6 + 4]*py0 + R[4*6 + 5]*de0 + R[4*6 + 6]*dl0
#     pos.dl = R[5*6 + 1]*rx0 + R[5*6 + 2]*px0 + R[5*6 + 3]*ry0 + R[5*6 + 4]*py0 + R[5*6 + 5]*de0 + R[5*6 + 6]*dl0
# end

function pm_identity_pass!(pos::Pos{T}, element::Element) where T
    return st_success
end

function pm_drift_pass!(pos::Pos{T}, element::Element) where T
    aux_drift(pos, element.properties[:length])
    return st_success
end

function pm_str_mpole_symplectic4_pass!(pos::Pos{T}, elem::Element, accelerator::Accelerator) where T
    # global_2_local(pos, elem)
    # steps = haskey(elem.properties, :nr_steps) ? elem.properties[:nr_steps] : 20
    steps = elem.properties[:nr_steps]


    sl = elem.properties[:length] / Float64(steps)
    l1 = sl * DRIFT1
    l2 = sl * DRIFT2
    k1 = sl * KICK1
    k2 = sl * KICK2
    polynom_b = elem.properties[:polynom_b]
    polynom_a = elem.properties[:polynom_a]
    rad_const = 0.0
    qexcit_const = 0.0

    if accelerator.radiation_state == radiation_damping
        rad_const = CGAMMA * (accelerator.energy/1e9)^3 / TWOPI
    end

    if accelerator.radiation_state == radiation_full
        qexcit_const = CQEXT * accelerator.energy^2 * sqrt(accelerator.energy * sl)
    end

    for i in 1:Int(steps)
        aux_drift(pos, l1)
        aux_strthinkick(pos, k1, polynom_a, polynom_b, rad_const, 0.0)
        aux_drift(pos, l2)
        aux_strthinkick(pos, k2, polynom_a, polynom_b, rad_const, qexcit_const)
        aux_drift(pos, l2)
        aux_strthinkick(pos, k1, polynom_a, polynom_b, rad_const, 0.0)
        aux_drift(pos, l1)
    end

    # local_2_global(pos, elem)
    return st_success
end

function pm_bnd_mpole_symplectic4_pass!(pos::Pos{T}, elem::Element, accelerator::Accelerator) where T
    # if !haskey(elem.properties, :angle)
    #     return pm_str_mpole_symplectic4_pass!(pos, elem, accelerator)
    # end
    #steps = haskey(elem.properties, :nr_steps) ? elem.properties[:nr_steps] : 20
    steps = elem.properties[:nr_steps]


    sl = elem.properties[:length] / Float64(steps)
    l1 = sl * DRIFT1
    l2 = sl * DRIFT2
    k1 = sl * KICK1
    k2 = sl * KICK2
    irho = elem.properties[:angle] / elem.properties[:length]

    polynom_b = elem.properties[:polynom_b]
    polynom_a = elem.properties[:polynom_a]
    
    rad_const = 0.0
    qexcit_const = 0.0

    if accelerator.radiation_state == radiation_damping
        rad_const = CGAMMA * (accelerator.energy / 1e9)^3 / (TWOPI)
    end

    if accelerator.radiation_state == radiation_full
        qexcit_const = CQEXT * accelerator.energy^2 * sqrt(accelerator.energy * sl)
    end

    ang_in = haskey(elem.properties, :angle_in) ? elem.properties[:angle_in] : 0.0
    ang_out = haskey(elem.properties, :angle_out) ? elem.properties[:angle_out] : 0.0
    fint_in = haskey(elem.properties, :fint_in) ? elem.properties[:fint_in] : 0.0
    fint_out = haskey(elem.properties, :fint_out) ? elem.properties[:fint_out] : 0.0
    gap = haskey(elem.properties, :gap) ? elem.properties[:gap] : 0.0


    #global_2_local(pos, elem)
    aux_edge_fringe(pos, irho, ang_in, fint_in, gap)

    for i in 1:1:Int(steps)
        aux_drift(pos, l1)
        aux_bndthinkick(pos, k1, polynom_a, polynom_b, irho, rad_const, 0.0)
        aux_drift(pos, l2)
        aux_bndthinkick(pos, k2, polynom_a, polynom_b, irho, rad_const, qexcit_const)
        aux_drift(pos, l2)
        aux_bndthinkick(pos, k1, polynom_a, polynom_b, irho, rad_const, 0.0)
        aux_drift(pos, l1)
    end

    aux_edge_fringe(pos, irho, ang_out, fint_out, gap)
    #local_2_global(pos, elem)

    return st_success
end

function pm_corrector_pass!(pos::Pos{T}, elem::Element) where T
    #global_2_local(pos, elem)
    xkick = elem.properties[:hkick]
    ykick = elem.properties[:vkick]

    if elem.properties[:length] == 0
        pos.px += hkick
        pos.py += vkick
    else
        rx, px = pos.rx, pos.px
        ry, py = pos.ry, pos.py
        de, dl = pos.de, pos.dl
        pnorm = 1 / (1 + de)
        norml = elem.properties[:length] * pnorm
        pos.dl += norml * pnorm * 0.5 * (xkick * xkick/3.0 + ykick * ykick/3.0 + px*px + py*py + px * xkick + py * ykick)
        pos.rx += norml * (px + 0.5 * xkick)
        pos.px += xkick
        pos.ry += norml * (py + 0.5 * ykick)
        pos.py += ykick
    end

    #local_2_global(pos, elem)
    return st_success
end

function pm_cavity_pass!(pos::Pos{T}, elem::Element, accelerator::Accelerator, turn_number::Int) where T
    if accelerator.cavity_state == off
        return pm_drift_pass!(pos, elem)
    end

    #global_2_local(pos, elem)
    nv = elem.properties[:voltage] / accelerator.energy
    philag = elem.properties[:phase_lag]
    frf = elem.properties[:frequency]
    harmonic_number = accelerator.harmonic_number
    velocity = light_speed
    #velocity = accelerator.velocity
    L0 = accelerator.length
    T0 = L0 / velocity

    if elem.properties[:length] == 0
        de, dl = pos.de, pos.dl
        #pos.de += -nv * sin(TWOPI * frf * (dl/velocity - (harmonic_number/frf - T0)*turn_number) - philag)
        pos.de += -nv * sin(TWOPI * frf * dl / velocity - philag)
    else
        rx, px = pos.rx, pos.px
        ry, py = pos.ry, pos.py
        de, dl = pos.de, pos.dl

        # Drift half length
        pnorm = 1 / (1 + de)
        norml = (0.5 * elem.properties[:length]) * pnorm
        pos.rx += norml * px
        pos.ry += norml * py
        pos.dl += 0.5 * norml * pnorm * (px*px + py*py)

        # Longitudinal momentum kick
        #pos.de += -nv * sin(TWOPI * frf * (dl/velocity - (harmonic_number/frf - T0)*turn_number) - philag)
        pos.de += -nv * sin(TWOPI * frf * dl / velocity - philag)

        # Drift half length
        pnorm = 1.0 / (1.0 + de)
        norml = (0.5 * elem.properties[:length]) * pnorm
        pos.rx += norml * px
        pos.ry += norml * py
        pos.dl += 0.5 * norml * pnorm * (px*px + py*py)
    end

    #local_2_global(pos, elem)
    return st_success
end
