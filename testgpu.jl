# # Packages
# using CUDA
# using Parameters
# using StaticArrays
# using Adapt
# using Setfield
# using LinearAlgebra
# using BenchmarkTools

# const T = Float32
# const ST = 6
# const NL = 1


# @with_kw struct Pos{T, ST}
#     N::Base.RefValue{Int}
#     a::AbstractVector{SVector{ST,T}} = zeros(SVector{ST,T}, NL[])
# end

# struct DevicePos{ST, T}
#     a::CuDeviceVector{SVector{ST,T}, 1}
# end

# Adapt.adapt_structure(to, s::Pos) = DevicePos(adapt(to, s.a))

# function gpu_func(V, leng)
#     index::Int = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride::Int = gridDim().x * blockDim().x
#     for i = index:stride:length(V.a)
#         pnorm::T = T(1 / (1 + V.a[i][5]))
#         norml::T = T(leng * pnorm)
#         s::SVector{ST, T} = SVector{ST, T}(
#             norml * V.a[i][2],
#             0,
#             norml * V.a[i][4],
#             0,
#             0,
#             0.5 * norml * pnorm * (V.a[i][2]*V.a[i][2] + V.a[i][4]*V.a[i][4])
#         )
#         @inbounds V.a[i] += s
#     end
#     return nothing
# end

# function kernel(V, leng)
#     kernel = @cuda launch=false gpu_func(V, leng)
#     config = launch_configuration(kernel.fun)
#     threads = min(length(V.a), config.threads)
#     blocks = cld(length(V.a), threads)
#     CUDA.@sync begin
#         kernel(V, leng; threads, blocks)
#     end
# end

# #ini = rand(SVector{ST, T}, NL)
# # ini = rand(SVector{ST, T}, NL)
# ini = [SVector{ST,T}(1,1,1,1,1,1) for _ in 1:1:NL]
# Vgpu = Pos{T, ST}(Ref(NL), CuArray(ini))
# l = 3.0f0
# @CUDA.time kernel(Vgpu, cu(l))

# ini = [SVector{ST,T}(1,1,1,1,1,1) for _ in 1:1:NL]
# Vgpu = Pos{T, ST}(Ref(NL), CuArray(ini))
# lc = cu(3.0f0)
# bench = @benchmark @CUDA.sync kernel($Vgpu, $lc)

# println("\n Benchmark GPU =\n")
# display(bench)

# using SiriusTrack.PosModule: Pos as SPos

# function drift(pos::SPos{Float64}, length::Float64) 
#     pnorm::Float64 = 1 / (1 + pos.de)
#     norml::Float64 = length * pnorm
#     pos.rx += norml * pos.px
#     pos.ry += norml * pos.py
#     pos.dl += 0.5 * norml * pnorm * (pos.px*pos.px + pos.py*pos.py)
# end

# p = SPos(ones(Float64, 6))
# drift(p, 3.0);

# lf = 3.0
# p = SPos(ones(Float64, 6))
# bc = @benchmark drift($p, $lf)

# println("\n Benchmark CPU =\n")
# display(bc)

using BenchmarkTools
using StaticArrays

const dim = 4 # 10_000_000

using CUDA

nthreads = CUDA.attribute(
    device(),
    CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
)

nblocks = cld(dim, nthreads)

# struct Pos{6, Float32}
#     N::Base.RefValue{Int}
#     parts::AbstractVector{SVector{6, Float32}} = zeros(SVector{6, Float32}, N[])
# end

# struct DevicePos{6, Float32}
#     parts::CuDeviceVector{SVector{6, Float32}, 1}
# end

# Adapt.adapt_structure(to, s::Pos) = DevicePos(adapt(to, s.a))

function drift_gpu_kernel!(V, leng)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(V)[2]
        #@inbounds z[i] = a * x[i] + y[i]
        @inbounds pnorm = 1f0 / (1f0 + V[5,i])
        norml = leng * pnorm
        @inbounds V[1,i] += (norml * V[2,i])
        @inbounds V[3,i] += (norml * V[4,i])
        @inbounds V[6,i] += (0.5f0 * norml * pnorm * (V[2,i]^2 + V[4,i]^2))
    end
    return nothing
end

const l0 = 3f0

p = CUDA.fill(1f0, (6, dim))

CUDA.@time CUDA.@sync @cuda(
    threads = nthreads,
    blocks = nblocks,
    drift_gpu_kernel!(p, l0)
)

p

# function drift_gpu_kernel!(rx, px, ry, py, de, dl, leng)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     if i <= length(rx)
#         #@inbounds z[i] = a * x[i] + y[i]
#         @inbounds pnorm = 1f0 / (1f0 + de[i])
#         norml = leng * pnorm
#         @inbounds rx[i] += norml #(norml[i] * px[i])
#         @inbounds ry[i] += (norml * py[i])
#         @inbounds dl[i] += (0.5f0 * norml * pnorm * (px[i]^2 + py[i]^2))
#     end
#     return nothing
# end
# CUDA.@sync @cuda(
#     threads = nthreads,
#     blocks = nblocks,
#     drift_gpu_kernel!(rx,px,ry,py,de,dl,l0)
# )
# Generate random data using CUDA.rand
x = CUDA.rand(Float32, (6, dim))

using SiriusTrack.PosModule: Pos

# Generate random Pos objects
pp = [Float32.(Pos(1)[:]) for _ in 1:dim]

# Convert pp to CuArray
xp = CUDA.zeros(Float32, (6, dim))

xp[:,:] = hcat(pp...)[:,:]

CUDA.@time CUDA.@sync @cuda(
    threads = nthreads,
    blocks = nblocks,
    drift_gpu_kernel!(xp, l0)
)

xp