# pos.jl

using Base

export Pos, pos, get_max

mutable struct Pos{Float64}
    rx::Float64
    px::Float64
    ry::Float64
    py::Float64
    de::Float64
    dl::Float64
end

pos(rx::Float64, px::Float64, ry::Float64, py::Float64, de::Float64, dl::Float64) = Pos(Float64(rx), Float64(px), Float64(ry), Float64(py), Float64(de), Float64(dl))

function Base.:+(v1::Pos{T}, v2::Pos{T}) where T
    return Pos{T}(v1.rx + v2.rx, v1.px + v2.px, v1.ry + v2.ry, v1.py + v2.py, v1.de + v2.de, v1.dl + v2.dl)
end

function Base.:+(m1::Vector{Pos{T}}, m2::Vector{Pos{T}}) where T
    return [m1[i] + m2[i] for i in 1:length(m1)]
end

function Base.:-(v1::Pos{T}, v2::Pos{T}) where T
    return Pos{T}(v1.rx - v2.rx, v1.px - v2.px, v1.ry - v2.ry, v1.py - v2.py, v1.de - v2.de, v1.dl - v2.dl)
end

function Base.:-(m1::Vector{Pos{T}}, m2::Vector{Pos{T}}) where T
    return [m1[i] - m2[i] for i in 1:length(m1)]
end

function  Base.:*(v::Pos{T}, scalar::S) where {T, S<:Real}
    return Pos{T}(v.rx * scalar, v.px * scalar, v.ry * scalar, v.py * scalar, v.de * scalar, v.dl * scalar)
end

function  Base.:*(scalar::S, v::Pos{T}) where {T, S<:Real}
    return Pos{T}(v.rx * scalar, v.px * scalar, v.ry * scalar, v.py * scalar, v.de * scalar, v.dl * scalar)
end

function  Base.:*(scalar::S, m1::Vector{Pos{T}}) where {T, S<:Real}
    return [scalar * v for v in m1]
end

function  Base.:/(v::Pos{T}, scalar::S) where {T, S<:Real}
    return (1 / scalar) * v
end

function Base.copy(p::Pos{T}) where T
    return Pos(p.rx, p.px, p.ry, p.py, p.de, p.dl)
end

function Base.show(io::IO, ::MIME"text/plain", p::Pos)
    println(io, "($(p.rx), $(p.px), $(p.ry), $(p.py), $(p.de), $(p.dl))")
end

# Revisar se eh necessario a redefinicao de "abs"
# function Base.:abs(x::Float64) where T
#     return ifelse(x > 0.0, x, -x)
# end

# function get_max(v::Pos{T}) where T
#     max_val = abs(v.rx)
#     max_val = max(abs(v.px), max_val)
#     max_val = max(abs(v.ry), max_val)
#     max_val = max(abs(v.py), max_val)
#     max_val = max(abs(v.de), max_val)
#     max_val = max(abs(v.dl), max_val)
#     return max_val
# end
