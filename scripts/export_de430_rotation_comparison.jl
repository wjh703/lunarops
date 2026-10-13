#= Export native DE430 rotation terms and a fully propagated reference arc.

Run with --project=../PlanetaryEphemeris.jl. The reference library is not
modified: a copy of its DE430! body has only its final return instrumented.
All physics, including the reference's Earth-figure conventions, is retained.
=#
using PlanetaryEphemeris
using TaylorIntegration
using TaylorSeries
using TOML

const PE = PlanetaryEphemeris
const N = length(PE.μ)
const JD0 = 2440400.5
const AU_M = PE.au * 1000
const DAY = PE.daysec
const DIAGNOSTIC_ORDER = 12

source_file = joinpath(pkgdir(PE), "src", "DE430", "dynamicalmodel.jl")
source = read(source_file, String)
start = findfirst("function DE430!(dq, q, params, t)", source)
body = source[first(start):end]
body = replace(body, "function DE430!" => "function de430_rotation_diagnostics!"; count=1)
return_terms = """
    return (; I_m_t, I_c_t, I_M_t, dI_m_t, q_del_τ_M, RotM,
        J2M_t, C21M_t, S21M_t, C22M_t, S22M_t,
        F_JCS_x, F_JCS_y, F_JCS_z,
        N_MfigM_pmA_x, N_MfigM_pmA_y, N_MfigM_pmA_z,
        earth_figure=[N_MfigM_figE_1, N_MfigM_figE_2, N_MfigM_figE_3],
        cmb=[N_cmb_1, N_cmb_2, N_cmb_3],
        inertia_rate_product=[dIω_x, dIω_y, dIω_z],
        mantle_cross=[ωxIω_x, ωxIω_y, ωxIω_z],
        latitude=sin_ϕ[ea,mo],
        earth_pole=[p_E_1,p_E_2,p_E_3], earth_direction=[er_EM_1,er_EM_2,er_EM_3])
end
"""
@assert endswith(strip(body), "nothing\nend")
body = replace(body, r"nothing\s*end\s*$" => return_terms)
Base.include_string(PE, body, source_file * " (diagnostics return only)")

function row(io, name, data)
    values = data isa Number ? [data] : (ndims(data) == 2 ? vec(permutedims(data)) : vec(data))
    println(io, name, '\t', join(repr.(Float64.(values)), '\t'))
end

function diagnostics(output, offset, q, params)
    println("Native component evaluation at $(offset) s")
    flush(stdout)
    t, x, dx = TaylorIntegration.init_expansions(offset / DAY, q, DIAGNOSTIC_ORDER)
    d = PE.de430_rotation_diagnostics!(dx, x, params, t)
    constant(v) = constant_term.(v)
    factor = PE.μ[PE.mo] * PE.R_moon^2
    delayed = constant(d.q_del_τ_M)
    r = delayed[3PE.ea-2:3PE.ea] - delayed[3PE.mo-2:3PE.mo]
    angles = delayed[6PE.N_bwd+1:6PE.N_bwd+3]
    rotation = PE.pole_rotation(angles[1] - pi/2, pi/2 - angles[2], angles[3])
    r_pa = rotation * r
    wm = delayed[6PE.N_bwd+4:6PE.N_bwd+6]
    open(joinpath(output, "terms_$(round(Int,offset)).tsv"), "w") do io
        row(io, "offset_s", offset)
        row(io, "positions_m", reshape(q[1:3N],3,N)[:,1:11]' .* AU_M)
        row(io, "velocities_mps", reshape(q[3N+1:6N],3,N)[:,1:11]' .* (AU_M/DAY))
        row(io, "euler", q[6N+1:6N+3])
        row(io, "wm", q[6N+4:6N+6] ./ DAY)
        row(io, "wc", q[6N+10:6N+12] ./ DAY)
        row(io, "delayed_q_native", delayed)
        row(io, "delayed_q_dot_native", [v[1] for v in d.q_del_τ_M])
        row(io, "undistorted", PE.ITM_und ./ factor)
        row(io, "core", constant(d.I_c_t) ./ factor)
        row(io, "mantle", constant(d.I_m_t) ./ factor)
        row(io, "total", constant(d.I_M_t) ./ factor)
        row(io, "tidal", PE.ITM1(r_pa...) ./ factor)
        row(io, "rotational", PE.ITM2(wm...) ./ factor)
        row(io, "inertia_dot", constant(d.dI_m_t) ./ (factor*DAY))
        row(io, "earth_matrix", constant(d.RotM[:,:,PE.ea])')
        row(io, "moon_matrix", constant(d.RotM[:,:,PE.mo])')
        row(io, "degree2_unnormalized", constant([ -d.J2M_t, d.C21M_t, d.S21M_t, d.C22M_t, d.S22M_t]) ./ PE.R_moon^2)
        sources = findall(PE.UJ_interaction[:,PE.mo])
        for i in sources
            f = constant([d.F_JCS_x[i,PE.mo],d.F_JCS_y[i,PE.mo],d.F_JCS_z[i,PE.mo]])
            torque = -constant(d.RotM[:,:,PE.mo]) * constant([d.N_MfigM_pmA_x[i],d.N_MfigM_pmA_y[i],d.N_MfigM_pmA_z[i]])
            row(io, "figure_acceleration_$(i)", f .* (PE.μ[PE.mo]*AU_M/DAY^2))
            row(io, "figure_torque_$(i)", torque ./ (PE.R_moon^2*DAY^2))
        end
        row(io, "earth_figure", constant(d.earth_figure) ./ (factor*DAY^2))
        row(io, "cmb", constant(d.cmb) ./ (factor*DAY^2))
        row(io, "inertia_rate_product", constant(d.inertia_rate_product) ./ (factor*DAY^2))
        row(io, "mantle_cross", constant(d.mantle_cross) ./ (factor*DAY^2))
        row(io, "reference_latitude_sine", constant_term(d.latitude))
        row(io, "earth_pole_pa", constant(d.earth_pole))
        row(io, "earth_direction_pa", constant(d.earth_direction))
        row(io, "euler_dot", constant(dx[6N+1:6N+3]) ./ DAY)
        row(io, "mantle_dot", constant(dx[6N+4:6N+6]) ./ DAY^2)
        row(io, "core_dot", constant(dx[6N+10:6N+12]) ./ DAY^2)
    end
    GC.gc()
end

function main()
    output = abspath(ARGS[1])
    duration = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 3.0
    mkpath(output)
    ic = joinpath(pkgdir(PE), "data", "de430ic_1969Jun28.txt")
    q0 = read_initial_conditions(ic)
    ids = [parse(Int, split(line)[1]) for line in readlines(ic)[2:12]]
    parameters = Dict(
        "jd0"=>JD0, "radius_m"=>PE.R_moon*AU_M, "love_k2"=>PE.k_2M,
        "delay_s"=>PE.τ_M*DAY, "mean_motion_rad_s"=>PE.n_moon/DAY,
        "core_fraction"=>PE.α_c, "core_flattening"=>PE.f_c,
        "viscosity_per_s"=>PE.k_ν/(PE.μ[PE.mo]*PE.R_moon^2*DAY),
        "earth_radius_m"=>PE.RE_au*AU_M, "earth_j2"=>PE.J2E,
        "earth_j2_rate_per_year"=>PE.J2EDOT,
        "body_ids"=>ids, "body_mu"=>PE.μ[1:11].*(AU_M^3/DAY^2),
        "source_indices"=>findall(PE.UJ_interaction[:,PE.mo]),
        "body_count"=>N, "diagnostic_order"=>DIAGNOSTIC_ORDER,
    )
    open(io -> TOML.print(io, parameters), joinpath(output,"parameters.toml"), "w")
    params = DE430Params(JD0, q0, DIAGNOSTIC_ORDER)
    diagnostics(output, 0.0, q0, params)
    duration == 0 && return
    order = 25
    println("Native full DE430 propagation: $(duration) days / $(N) bodies / $(Threads.nthreads()) threads")
    flush(stdout)
    problem = PlanetaryEphemerisProblem(DE430!, (JD0, JD0+duration), q0, DE430Params(JD0,q0,order))
    @time solution = propagate(problem; maxsteps=100000,order,abstol=1e-20,parse_eqs=true)
    state_at(s) = solution((JD0-PE.J2000)+s/DAY)
    open(joinpath(output,"trajectory.tsv"),"w") do io
        for offset in 0.0:300.0:(duration*DAY)
            q = state_at(offset)
            row(io, "sample", vcat(offset, reshape(q[1:3N],3,N)[:,1:11]'[:] .* AU_M,
                reshape(q[3N+1:6N],3,N)[:,1:11]'[:] .* (AU_M/DAY),
                q[6N+1:6N+3],q[6N+4:6N+6]./DAY,q[6N+10:6N+12]./DAY))
        end
    end
    for offset in unique([DAY,duration*DAY])
        diagnostics(output,offset,state_at(offset),params)
    end
    println("Reference exported to $output")
end
main()
