# Generate collocation nodes and quadrature weights
function generate_collocation_nodes(param_intervals::Dict{Num, IntervalArithmetic.Interval{Float64}}, poly_order::Int)
    param_array = collect(param_intervals)
    d = length(param_array)

    N = poly_order+1
    xi_1D, _ = gausslegendre(N)

    scaled_nodes = Vector{Vector{Float64}}(undef, d)
    xi_nodes = Vector{Vector{Float64}}(undef, d)
    param_keys = Vector{Any}(undef, d)

    for j in 1:d
        (param_key, interval) = param_array[j]
        a, b = inf(interval), sup(interval)
        scaled_nodes[j] = (xi_1D .+ 1) .* ((b - a)/2) .+ a
        xi_nodes[j] = xi_1D
        param_keys[j] = param_key
    end

    scaled_product = collect(product(scaled_nodes...))
    xi_product = collect(product(xi_nodes...))

    total_points = length(scaled_product)
    collocation_nodes = Vector{Dict{Any, Float64}}(undef, total_points)
    xi_combinations = Vector{Vector{Float64}}(undef, total_points)

    for i in 1:total_points
        tmp_dict = Dict{Any, Float64}()
        tmp_xi = Vector{Float64}(undef, d)

        for j in 1:d
            tmp_dict[param_keys[j]] = scaled_product[i][j]
            tmp_xi[j] = xi_product[i][j]
        end

        collocation_nodes[i] = tmp_dict
        xi_combinations[i] = tmp_xi
    end

    return collocation_nodes, xi_combinations
end

function generate_quadrature_weights(d::Int, poly_order::Int)
    N = poly_order + 1
    _, weights_1D = gausslegendre(N)
    weights_product = collect(product((weights_1D for _ in 1:d)...))
    quadrature_weights = [prod(tup) for tup in weights_product]
    return quadrature_weights
end

# Validate that interesting variables aren't parameters
function validate_interesting_variables(sys, interesting_vars)
    all_params = ModelingToolkit.parameters(sys)
    common_params = filter(x -> any(y -> isequal(x, y), all_params), interesting_vars)
    if !isempty(common_params)
        error("Interesting variables cannot be parameters. Problematic variables: ", common_params)
    end
end

function legendre_polynomials(ξ)
    return [
        1.0,
        ξ,
        0.5*(3ξ^2 - 1),
        0.5*(5ξ^3 - 3ξ),
        (1/8)*(35ξ^4 - 30ξ^2 + 3),
        (1/8)*(63ξ^5 - 70ξ^3 + 15ξ),
        (1/16)*(231ξ^6 - 315ξ^4 + 105ξ^2 - 5),
        (1/16)*(429ξ^7 - 693ξ^5 + 315ξ^3 - 35ξ),
        (1/128)*(6435ξ^8 - 12012ξ^6 + 6930ξ^4 - 1260ξ^2 + 35),
        (1/128)*(12155ξ^9 - 25740ξ^7 + 18018ξ^5 - 4620ξ^3 + 315ξ),
        (1/256)*(46189ξ^10 - 109395ξ^8 + 90090ξ^6 - 30030ξ^4 + 3465ξ^2 - 63)
    ]
end

# Generate total-degree multi-indices
function generate_total_degree_multi_indices(d, p)
    return [i for i in Iterators.product((0:p for _ in 1:d)...)if sum(i) <= p]
end

# Main function for Polynomial Chaos Expansion interval analysis
function run_pce_interval_analysis(sys, dim, tspan, dt, param_intervals, interesting_vars, verbose, solver)
    if dim > 10
        error("Dimensions above 10 are not supported.")
    end

    validate_interesting_variables(sys, interesting_vars)

    
    d = length(param_intervals)
    poly_order = 2

    collocation_nodes, xi_nodes = generate_collocation_nodes(param_intervals, poly_order)
    
    quadrature_weights = generate_quadrature_weights(d, poly_order)
    #quadrature_weights= 0.5^d .* quadrature_weights
    #println(collocation_nodes)
    #println(length(collocation_nodes))
    save_times = tspan[1]:dt:tspan[2]
    N_nodes = length(collocation_nodes)
    solutions = Vector{ODESolution}(undef, N_nodes)

    all_states = unknowns(sys)
    #println(all_states)
    p_dict = copy(collocation_nodes[1])
    ic_dict = Dict{Num, Float64}()
    for u in all_states
        if haskey(p_dict, u)
            ic_dict[u] = p_dict[u]
            delete!(p_dict, u)
        end
    end
    #println(p_dict)
    prob = ODEProblem(sys, ic_dict, (tspan[1], tspan[2]), p_dict)
    setp! = ModelingToolkit.setp(prob, [keys(p_dict)...])
    setu! = ModelingToolkit.setu(prob, [keys(ic_dict)...])

    if verbose
        println("Creating ", N_nodes, " problems to solve")
    end

    for i in ProgressBar(1:N_nodes)
        p_dict = copy(collocation_nodes[i])
        ic_dict = Dict{Num, Float64}()
        for u in all_states
            if haskey(p_dict, u)
                ic_dict[u] = p_dict[u]
                delete!(p_dict, u)
            end
        end
        #setp!(prob, values(p_dict))
        #setu!(prob, values(ic_dict))
        prob = ODEProblem(sys, ic_dict, (tspan[1], tspan[2]), p_dict)

        #println("")
        #println(p_dict)
        #println("")
        solutions[i] = solve(prob, solver, saveat=save_times)
    end

    for k in 1:N_nodes
        #println(solutions[k][10])
        if solutions[k].retcode != :Success
            error("Solver failed for collocation index $k. retcode=$(solutions[k].retcode)")
        end
    end
    

    if verbose
        println("Done solving, calculating intervals")
    end

    # Gives all indeces to all posible combinations of parameters in the grid
    #multi_indices = collect(product((0:poly_order for _ in 1:d)...))
    #M = length(multi_indices)

    multi_indices = generate_total_degree_multi_indices(d, poly_order)
    M = length(multi_indices)
    
    # Precompute polynomial basis
    #Exactly like equation 22, 
    #phi_table = zeros(N_nodes, M)
    phi_table = Matrix{Float64}(undef, N_nodes, M)
    #println(phi_table)
    @threads for k in 1:N_nodes
        ξs = xi_nodes[k]
        #println(ξs)
        for m_idx in 1:M
            multi_idx = multi_indices[m_idx]
            val = prod(legendre_polynomials(ξs[dim_i])[multi_idx[dim_i]+1] for dim_i in 1:d)
            phi_table[k,m_idx] = val
        end
    end

    # Normalization factors
    norm_factors = zeros(M)
    println("")
    println(multi_indices[2])
    @threads for m_idx in 1:M
        multi_idx = multi_indices[m_idx]
        norm_factors[m_idx] = prod(((2*i + 1)/2) for i in multi_idx) #From equation 31 together with 18
    end

    # Precompute polynomial interval bounds
    legendre_intervals = [interval(1.0), interval(-1.0, 1.0), interval(-0.5,1.0), interval(-1.0,1.0),
                          interval(-0.4286,1.0), interval(-1.0,1.0), interval(-0.4147,1.0),
                          interval(-1.0,1.0), interval(-0.4097,1.0), interval(-1.0,1.0),
                          interval(-0.4073,1.0)]
    println(multi_indices)
    poly_interval = [prod(legendre_intervals[i+1] for i in multi_idx) for multi_idx in multi_indices] ## Här borde ju bli wrapping? Borde göra något mer avancerat för att bli av med det?
    #Eventuellt faktiskt gör symboliska beräkningar på legendre polynomen och sen hitta max och min på intervallet?
    #println(phi_table[1,10])
    println(poly_interval)
    poly_interval_map = Dict{NTuple{d, Int}, IntervalArithmetic.Interval{Float64}}()
    # Degree 0
    poly_interval_map[(0, 0)] = interval(1.0)

    # Degree 1
    poly_interval_map[(1, 0)] = interval(-1.0, 1.0)
    poly_interval_map[(0, 1)] = interval(-1.0, 1.0)

    # Degree 2
    poly_interval_map[(2, 0)] = interval(-0.5, 1.0)
    poly_interval_map[(0, 2)] = interval(-0.5, 1.0)
    poly_interval_map[(1, 1)] = interval(0.0, 1.0)

    # Degree 3
    poly_interval_map[(3, 0)] = interval(-1.0, 1.0)
    poly_interval_map[(0, 3)] = interval(-1.0, 1.0)
    poly_interval_map[(2, 1)] = interval(-1.0, 1.0)
    poly_interval_map[(1, 2)] = interval(-1.0, 1.0)

    # Degree 4
    poly_interval_map[(4, 0)] = interval(-0.438571, 1.0)
    poly_interval_map[(0, 4)] = interval(-0.438571, 1.0)
    poly_interval_map[(3, 1)] = interval(-0.225, 1.0)  
    poly_interval_map[(1, 3)] = interval(-0.225, 1.0)  
    poly_interval_map[(2, 2)] = interval(0.0, 1.0)    

    #Degree 5
    poly_interval_map[(2, 3)] = interval(-1.0, 1.0)
    poly_interval_map[(3, 2)] = interval(-1.0, 1.0)
    poly_interval_map[(3, 3)] = interval(0.0, 1.0)  
    # ... and so on

    ## Så även om xi_s 'r oberoende. So  

    zero_multi_idx = ntuple(_->0, d)

    unks = Tuple(vcat(unknowns(sys), interesting_vars))
    num_times = length(save_times)
    results = Dict{Any, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()

    # We only consider one unknown at the time, 
    for unk in ProgressBar(unks)
        coeff_dict = Dict(multi_idx => zeros(num_times) for multi_idx in multi_indices)
        mean_vals, lower_vals, upper_vals = zeros(num_times), zeros(num_times), zeros(num_times)

        for t_idx in 1:num_times
            y_vals = [solutions[k][unk][t_idx] for k in 1:N_nodes]
            enclosure = interval(0.0,0.0)

            @threads for m_idx in 1:M
                local_sum = sum(quadrature_weights[k]*y_vals[k]*phi_table[k,m_idx] for k in 1:N_nodes) # Equation 34 in Wang and Yang
                coeff_dict[multi_indices[m_idx]][t_idx] = norm_factors[m_idx]*local_sum # Equation 30 (norm_factors are 1/denominator)
            end

            for m_idx in 1:M
                multi_idx = multi_indices[m_idx]
                enclosure += coeff_dict[multi_idx][t_idx] * poly_interval[m_idx]# poly_interval_map[multi_idx]
            end

            lower_vals[t_idx] = inf(enclosure)
            upper_vals[t_idx] = sup(enclosure)
            mean_vals[t_idx] = coeff_dict[zero_multi_idx][t_idx]
        end
        results[unk] = (mean_vals, lower_vals, upper_vals)
    end
    
    return results, save_times
end