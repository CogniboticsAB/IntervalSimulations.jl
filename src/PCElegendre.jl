
################################################################################
#### 1) BUILD PARAMETER COMBINATIONS, WEIGHTS  ####
################################################################################

function build_parameter_combinations(param_dict::Dict{Num, IntervalArithmetic.Interval{Float64}}, p::Int)
    # 1) Collect the uncertain parameters into an array of (key, interval) pairs.
    param_array = collect(param_dict)  # e.g. [(:l1, 0.9..1.1), (:l2, 0.9..1.1)]
    d = length(param_array)            # number of uncertain parameters

    # 2) Gauss–Legendre on [-1, 1] for order p => (p+1) nodes
    N = p + 1
    nodes_1D, _ = gausslegendre(N)

    # 3) For each parameter j, build arrays of scaled and unscaled nodes
    mapped_node_arrays   = Vector{Vector{Float64}}(undef, d)
    unscaled_node_arrays = Vector{Vector{Float64}}(undef, d)
    keys_array           = Vector{Any}(undef, d)

    for j in 1:d
        (param_key, iv) = param_array[j]
        a, b = inf(iv), sup(iv)
        # Map [ -1,1 ] -> [ a,b ]
        mapped_node_arrays[j] = (nodes_1D .+ 1) .* ((b - a)/2) .+ a
        unscaled_node_arrays[j] = nodes_1D
        keys_array[j] = param_key
    end

    # 4) Build Cartesian product of mapped nodes (scaled) and unscaled
    mapped_product   = collect(product(mapped_node_arrays...))
    unscaled_product = collect(product(unscaled_node_arrays...))
    # Each has length (p+1)^d
    total_points = length(mapped_product)

    # 5) Convert each product element into the desired form:
    #    - Dict of scaled parameter values
    #    - Vector of unscaled (xi) values
    param_combinations    = Vector{Dict{Any,Float64}}(undef, total_points)
    unscaled_combinations = Vector{Vector{Float64}}(undef, total_points)

    for i in 1:total_points
        mapped_tuple   = mapped_product[i]
        unscaled_tuple = unscaled_product[i]

        # Build the dict of scaled values
        tmp_dict = Dict{Any,Float64}()
        tmp_xi   = Vector{Float64}(undef, d)

        for j in 1:d
            tmp_dict[keys_array[j]] = mapped_tuple[j]
            tmp_xi[j]              = unscaled_tuple[j]
        end

        param_combinations[i]    = tmp_dict
        unscaled_combinations[i] = tmp_xi
    end

    return param_combinations, unscaled_combinations
end


function build_multi_weights(d::Int, p::Int)
    # For each dimension, get the same 1D weights from Gauss–Legendre on [-1,1].
    N = p + 1
    _, w_1D = gausslegendre(N)

    # Build the Cartesian product of these d weight arrays
    w_product = collect(product((w_1D for _ in 1:d)...))

    # w_combined[k] = product of the d 1D weights in w_product[k]
    w_combined = [prod(tup) for tup in w_product]
    return w_combined
end

function checkForParameters(fol,intresting_variables)
    # Get all parameters from the system
    all_params = ModelingToolkit.parameters(fol)

    # Check if any interesting variable matches a parameter using isequal explicitly
    common_params = filter(x -> any(y -> isequal(x, y), all_params), intresting_variables)

    # Output result
    if !isempty(common_params)
        error("Cant have parameters as intresting variables, they are simply not intresting 🐱! Problem with ", common_params)
    end
end


################################################################################
#### 2) MAIN 'run' FUNCTION  ####
################################################################################

function runlegendre(sys, dim, ts, dt, pval, intresting_variables, print)
    if dim > 10
        error("Dimensions above 10 are not supported.")
    end

    checkForParameters(sys,intresting_variables)

    p = dim
    d = length(pval)

    # 1) Quadrature points (scaled -> param_combos, and unscaled -> nodes)
    param_combos, nodes = build_parameter_combinations(pval, p)

    # 2) Build product weights (without the 1/2 factor). We'll multiply that later.
    base_weights = build_multi_weights(d, p)

    # 3) Solve ODE for each quadrature node
    save_times = ts[1]:dt:ts[2]
    Ngrid = length(param_combos)

    solutions = Vector{ODESolution}(undef, Ngrid)
    # Get the system’s state variables (the unknowns)
    # Assume uval is a dictionary mapping each state variable (unknown) to its default initial condition.
    all_states = unknowns(sys)  # the system’s state variables    

    #Initialize a standard problem to use for remakes
    p_dict = copy(param_combos[1])
    ic_dict = Dict{Num,Float64}()
    for u in all_states
        if haskey(p_dict, u)
            ic_dict[u] = p_dict[u]
            delete!(p_dict, u)
        end
    end
    prob = ODEProblem(sys, ic_dict, (ts[1], ts[2]), p_dict)

    setp! = ModelingToolkit.setp(prob, [keys(p_dict)...])
    setu! = ModelingToolkit.setu(prob, [keys(ic_dict)...])
    ###########################################################
    
    if print
        println("Creating ",Ngrid," probs to solve")
    end
    a= true
    for i in ProgressBar(1:Ngrid)
        p_dict = copy(param_combos[i])
        ic_dict = Dict{Num,Float64}()
        for u in all_states
            if haskey(p_dict, u)
                ic_dict[u] = p_dict[u]
                delete!(p_dict, u)
            end
        end
        #ic = collect(ic_dict)
        # If you pass fol, you won't re-trigger codegen each time:
        #prob = remake(prob, u0=ic, p=p_dict)
        #problems[i] = prob
        setp!(prob,values(p_dict))
        setu!(prob,values(ic_dict))
        #println(prob.ps[sys.dynamics.amplitude])
        solutions[i] = solve(prob, saveat=save_times)
    end
    
    # 3) Solve the pre-built problems in parallel:
    #solutions = Vector{ODESolution}(undef, Ngrid)


  # if print
  #     println("Solving ", Ngrid, " problems")
  # end
  # @time for i in ProgressBar(1:Ngrid)
  #     solutions[i] = solve(problems[i], saveat=save_times)
  # end

    if print
        println("Done solving, calculating intervals")
    end
    #return solutions

    # Standard Legendre polynomials P₀..P₁₀ on [-1,1]
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

    # We'll multiply Gauss–Legendre weights by 1/2 in each dimension => (1/2)^d
    # to reflect the uniform distribution pdf on [-1,1]^d.
    #prob_weights = 0.5^d .* base_weights

    # Precompute all possible multi-indices
    all_i_tuples = collect(product((0:p for _ in 1:d)...))
    # e.g. if d=2, p=2 => (0,0), (0,1), (0,2), (1,0), etc.

    # -------------------------------------------------------------------------
    # (1) PRECOMPUTE the polynomial products for each node, multi-index
    #     "legendre_prod(nodes[k], iTup)" so we don't repeat them every time step.
    # -------------------------------------------------------------------------
    # M = number of multi-indices = (p+1)^d
    M = length(all_i_tuples)
    # We'll store phi_table[k, m] = legendre_prod( nodes[k], all_i_tuples[m] )
    # i.e. the product of P_{i_j}(ξ_{k,j}) for j=1..d.
    phi_table = zeros(Ngrid, M)

    # Make a local function that returns Pᵢ(ξ):
    # so we only compute the polynomials up to p each time.
    function poly_at(x, i)
        # i in 0..p
        return legendre_polynomials(x)[i + 1]
    end

    @threads for k in 1:Ngrid
        ξs = nodes[k]  # each is length d
        for m_idx in 1:M
            iTup = all_i_tuples[m_idx]
            val = 1.0
            for dim_i in 1:d
                val *= poly_at(ξs[dim_i], iTup[dim_i])
            end
            phi_table[k, m_idx] = val
        end
    end

    # -------------------------------------------------------------------------
    # (2) PRECOMPUTE the "normalization" factor for each multi-index
    #     norm_factor(iTup) = ∏( (2*i + 1)/2 ).
    # -------------------------------------------------------------------------
    # This is the same logic your loop does, just stored in a table/dict up front.
    norm_factors = zeros(M)
    @threads for m_idx in 1:M
        iTup = all_i_tuples[m_idx]
        nf = 1.0
        for idx in iTup
            nf *= (2*idx + 1)/2
        end
        norm_factors[m_idx] = nf
    end

    # -------------------------------------------------------------------------
    # (3) PRECOMPUTE the bounding interval for each multi-index
    #     by multiplying the known intervals of each Pᵢ(ξ).
    # -------------------------------------------------------------------------
    legendre_intervals = [
        interval(1.0, 1.0),        # P₀
        interval(-1.0, 1.0),       # P₁
        interval(-0.5, 1.0),       # P₂
        interval(-1.0, 1.0),       # P₃
        interval(-0.4286, 1.0),    # P₄
        interval(-1.0, 1.0),       # P₅
        interval(-0.4147, 1.0),    # P₆
        interval(-1.0, 1.0),       # P₇
        interval(-0.4097, 1.0),    # P₈
        interval(-1.0, 1.0),       # P₉
        interval(-0.4073, 1.0)     # P₁₀
    ]
    # We'll store poly_interval[m_idx] = product of intervals for that iTup
    poly_interval = Vector{IntervalArithmetic.Interval{Float64}}(undef, M)
    for m_idx in 1:M
        iTup = all_i_tuples[m_idx]
        r = interval(1.0, 1.0)
        for idx in iTup
            r *= legendre_intervals[idx + 1]
        end
        poly_interval[m_idx] = r
    end

    # We'll also identify the zero-tuple to pick out the mean easily:
    zeroTup = ntuple(_->0, d)
    zeroIdx = findfirst(==(zeroTup), all_i_tuples)

    # -------------------------------------------------------------------------
    # Now we do the same logic as your original loops, but we reuse phi_table,
    # norm_factors, poly_interval, etc. so we don't recompute them inside the loops.
    # -------------------------------------------------------------------------
    unk = unknowns(sys)
    unks=Tuple([unk...,intresting_variables...])
    num_times = length(save_times)
    results = Dict{Any,Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()

    # A place to store the partial sums for each iTup
    # (We still use the same `coeff_dict[iTup] = zeros(num_times)` logic.)
    count = 0
    for unk in ProgressBar(unks)
        coeff_dict = Dict{NTuple{d,Int},Vector{Float64}}()
        for iTup in all_i_tuples
            coeff_dict[iTup] = zeros(num_times)
        end

        # We'll also store the final mean/lower/upper:
        mean_vals  = zeros(num_times)
        lower_vals = zeros(num_times)
        upper_vals = zeros(num_times)

        for t_idx in 1:num_times
            # 1) Gather the solution at each grid point
            #    => y_vals[k] = solutions[k][unk][t_idx]
            y_vals = [solutions[k][unk][t_idx] for k in 1:Ngrid]

            enclosure = interval(0.0, 0.0)

            # 2) For each multi-index, compute the integral approximation
            @threads for m_idx in 1:M
                iTup = all_i_tuples[m_idx]
                # local_sum = Σ ( base_weights[k] * y_vals[k] * phi_table[k,m_idx] )
                # then multiply by norm_factors[m_idx].
                # Note that if you want the measure to be uniform, you can do
                # prob_weights instead of base_weights. But your code uses base_weights
                # *and then multiplies by 1/2 in the final "IMPORTANT" comment. 
                # We'll keep it exactly as your code does: "include factor (1/2)"
                # after this sum. So let's do that:
                local_sum = 0.0

                #Don't put threads here, much slower
                for k in 1:Ngrid
                    local_sum += base_weights[k] * y_vals[k] * phi_table[k,m_idx]
                end
                coeff_dict[iTup][t_idx] = norm_factors[m_idx] * local_sum
            end

            # sum up intervals
            
            for m_idx in 1:M
                iTup = all_i_tuples[m_idx]
                a_i = coeff_dict[iTup][t_idx]
                enclosure += a_i * poly_interval[m_idx] 
            end
            lower_vals[t_idx] = inf(enclosure)
            upper_vals[t_idx] = sup(enclosure)
            # mean is the coefficient of the all-zero multi-index
            mean_vals[t_idx] = coeff_dict[zeroTup][t_idx]
            #println(100*(count*num_times+t_idx)/(length(unks)*num_times), "%")
        end

        # Now build mean/lower/upper from these coefficients:
        # (Your code does it in the same "for t_idx in 1:num_times" loop,
        # but let's do it afterward for clarity.  The final result is the same.)
        results[unk] = (mean_vals, lower_vals, upper_vals)
        #count = count + 1
        #if print
        #    println(100*count/length(unks),"%")
        #end
    end

    return results, save_times
end
