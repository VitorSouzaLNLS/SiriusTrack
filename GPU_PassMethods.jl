
using CUDA

const dim = 10_000_000

nthreads = CUDA.attribute(
    device(),
    CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
)

nblocks = cld(dim, nthreads)

# V is a CUDA Matrix, with dimensions: (6, nr_electrons)

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

# all have dim = nr_particles, except curv (is Float (32))
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
            dl_ds = 1.0 + ((px*px + py*py) / 2)
            V[5,i] -= rad_const * delta_factor * b2p * dl_ds * length
        
            if qexit_const != 0f0
                d = delta_factor * qexit_const * sqrt(b2p^1.5 * dl_ds)
                V[5,i] += d * randn(Float32)
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
            delta_factor = (1f0 + V[5,i])^2
            dl_ds = 1.0 + ((px*px + py*py) / 2)
            V[5,i] -= rad_const * delta_factor * b2p * dl_ds * length
        
            if qexit_const != 0f0
                d = delta_factor * qexit_const * sqrt(b2p^1.5 * dl_ds)
                V[5,i] += d * randn(Float32)
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

# x = CUDA.rand(Float32, (6, dim))

x = CUDA.ones(Float32, (6, dim))

pB = CUDA.rand(Float32, 5)

pA = CUDA.zeros(Float32, 5)

leng = 0.2f0

CUDA.@time CUDA.@sync @cuda(
    threads = nthreads,
    blocks = nblocks,
    #_gpu_b2_perp(b2p, isum, rsum, x[2,:]*leng, x[4,:]*leng)
    _gpu_strthinkick(x, leng, pA, pB, 0.001f0, 0f0)
)