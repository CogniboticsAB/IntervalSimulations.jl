function check_initial_conditions(sys)
    # Extract unknowns from the system
    var_order = unknowns(sys)

    df1 = defaults(sys)                               # a Dict
    df2 = ModelingToolkit.missing_variable_defaults(sys)  # a Vector of Pairs
    if !isempty(df2)
       @warn "Missing initial values, $df2 is applied"
    end
    # Get default initial values
    default_values = merge(Dict(df1), Dict(df2))

    # Find missing initial conditions
    missing_vars = [var for var in var_order if !haskey(default_values, var)]

    # Throw an error if any initial conditions are missing
    if !isempty(missing_vars)
        error("Missing initial conditions for the following variables: $(missing_vars). Please define them in `defaults(sys)` or explicitly provide them in the `ODEProblem` call.")
    end
end

# A version of resolve_equations that expects a vector of equations.
function resolve_equations(eqs, resolved_defaults; inf_replacement=1e9)
    # Build a mapping from each LHS to its RHS.
    eq_map = Dict{Any,Any}()
    for eq in eqs
        eq_map[eq.lhs] = eq.rhs
    end
    def = resolved_defaults

    # We'll collect unresolved unknowns here.
    unresolved_vars = Any[]

    # This helper recursively resolves an expression 'v'.
    function resolve_value(v)
        # If v is not a symbolic expression, return it (replace Inf if needed)
        #println(typeof(v))
        if !(v isa SymbolicUtils.BasicSymbolic{Real} || v isa SymbolicUtils.BasicSymbolic{Float64}|| v isa SymbolicUtils.BasicSymbolic{Rational{Int64}}|| v isa SymbolicUtils.BasicSymbolic{Int64})
            #println(v)

            return v == Inf ? inf_replacement : v
        end

        # If v is a single variable (i.e. not a tree) and it appears in our mapping,
        # then substitute it by recursively resolving the corresponding RHS.
        if !istree(v) && haskey(def, v)
            # If the default value for v is an interval, we want to keep v (so that its uncertainty is preserved)
            if def[v] isa IntervalArithmetic.Interval
                if !any(u -> isequal(u, v), unresolved_vars) 
                    push!(unresolved_vars, v)
                end

                return v
            end
            return resolve_value(def[v])
        end

        # If v is a tree (an operation with arguments), resolve each argument
        # and then rebuild the expression.
        if istree(v)
            new_args = [resolve_value(arg) for arg in arguments(v)]
            result = operation(v)(new_args...)
            return result
        end

        # Default: we reached a symbol (an unknown variable) that could not be further resolved.
        # Record it in the unresolved_vars array and return it.
        return v
    end

    # Now, for each equation in eqs, resolve its RHS and store the result keyed by its LHS.
    resolved = Dict{Any,Any}()
    for eq in eqs
        resolved[eq.lhs] = resolve_value(eq.rhs)
    end
    return resolved, unresolved_vars
end


function generate_compiled_functions_with_baked_params(model, resolved_eqs, unresolved_vars)
    # Create a full list of state variables (the original unknowns plus the promoted ones)

    full_vars = [unknowns(model)...; unresolved_vars...]
    # We'll pass [full_vars..., t] to build_function:
    all_args = [full_vars...; ModelingToolkit.t]

    # Preallocate a vector to hold the compiled functions.
    compiled_functions = Vector{Any}(undef, length(full_vars))

    # For each variable in the full set, if it is in unresolved_vars, assign a zero derivative;
    # otherwise, try to build its derivative function from resolved_eqs.
    for (i, var) in enumerate(full_vars)
        if any(u -> isequal(u, var), unresolved_vars)
            # For the newly promoted variable, its derivative should be zero.
            compiled_functions[i] = Symbolics.build_function(0, all_args...; expression=false)
        else
            # For the original unknowns, build the derivative function.
            key = Differential(t)(var)
            if haskey(resolved_eqs, key)
                rhs = resolved_eqs[key]
                compiled_functions[i] = Symbolics.build_function(rhs, all_args...; expression=false)
            else
                # If no differential equation is given, default to zero.
                compiled_functions[i] = Symbolics.build_function(0.0, all_args...; expression=false)
            end
        end
    end

    return compiled_functions
end


function resolve_defaults(dict; inf_replacement=1e9, model=nothing)
    # First pass: recursively resolve the dictionary entries as you already do.
    partially_resolved = Dict{Any,Any}()
    
    function local_resolve(v)
        if !(v isa SymbolicUtils.BasicSymbolic{Real})
            return v == Inf ? inf_replacement : v
        end
        if !istree(v) && haskey(dict, v)
            return local_resolve(dict[v])
        end
        if istree(v)
            new_args = [local_resolve(arg) for arg in arguments(v)]
            return operation(v)(new_args...)
        end
        return v
    end

    for (k, v) in dict
        partially_resolved[k] = local_resolve(v)
    end

    # If a model is provided and it has observed equations, try to fill in missing defaults.
    if model !== nothing
        for eq in observed(model)
            # We assume an equation of the form lhs ~ rhs.
            # If one side is not in the defaults but the other is a number, add it.
            if !haskey(partially_resolved, eq.lhs) && haskey(partially_resolved, eq.rhs)
                v = partially_resolved[eq.rhs]
                if v isa Number
                    partially_resolved[eq.lhs] = v
                end
            elseif !haskey(partially_resolved, eq.rhs) && haskey(partially_resolved, eq.lhs)
                v = partially_resolved[eq.lhs]
                if v isa Number
                    partially_resolved[eq.rhs] = v
                end
            end
        end
    end

    # Second pass: build a substitution map for keys that are fully resolved (numbers).
    subs_map = Dict{Any,Any}()
    for (k, v) in partially_resolved
        if v isa Number
            subs_map[k] = v
        end
    end

    # Third pass: perform a full substitution on each expression.
    fully_resolved = Dict{Any,Any}()
    for (k, v) in partially_resolved
        if v isa SymbolicUtils.BasicSymbolic
            fully_resolved[k] = Symbolics.substitute(v, subs_map)
        else
            fully_resolved[k] = v
        end
    end

    return fully_resolved
end

function createIVP(fol)
    # Run on your defaults dictionary

    #ode=ODEProblem(fol)
    #fol=modelingtoolkitize(ode)

    if has_alg_equations(fol)
        #throw("Has algrebraic equation, not supported atm.")
        eqs = equations(fol)

        # 3) Solve algebraic constraint (assumed to be the last eq, not right)
        eq_constraint = eqs[end]  
        expr = eq_constraint.rhs
        solution = symbolic_linear_solve(expr, unknowns(fol)[end])
        expr_for_w = solution   

        # 4) Substitute into the other equations (1:4), dropping the last
        eqs_original = eqs[1:end-1]
        sub_map = Dict(unknowns(fol)[end] => expr_for_w)
        eqs_substituted = substitute.(eqs_original, Ref(sub_map))

        # Get the list of parameters
        params = ModelingToolkit.parameters(fol)

        # 6) Make an ODE system
        @named minimal_sys = ODESystem(eqs_substituted, t, unknowns(fol)[1:end-1], params, tspan=(0,10);)

        # (Optional) further simplification
        #fol= structural_simplify(minimal_sys)
        fol = minimal_sys
    end

    check_initial_conditions(fol)

    df1 = defaults(fol)                               # a Dict
    df2 = ModelingToolkit.missing_variable_defaults(fol)  # a Vector of Pairs
    
    # Get default initial values
    default_values = merge(Dict(df1), Dict(df2))

    resolved_defaults = resolve_defaults(default_values, model=fol)

    eqs, unresolved_vars = resolve_equations(full_equations(fol), resolved_defaults)
    
    compiled_functions= generate_compiled_functions_with_baked_params(fol, eqs, unresolved_vars)
    #return compiled_functions
    # Extract initial values and construct IntervalBox
    var_order = unknowns(fol);
    append!(var_order, unresolved_vars);

    initial_values = [resolved_defaults[var] for var in var_order];
    X₀ = IntervalBox(initial_values...);#, init_p...)
    function fol!(du, x, p, t)
        nvars = length(X₀);
        # Evaluate each derivative in order.
        for i in 1:nvars
            du[i] = compiled_functions[i](x[1:nvars]..., t) + zero(x[i]);
        end
    end

    # Define and solve the IVP
    order = length(X₀);
    prob = @ivp(x' = fol!(x), dim:order, x(0) ∈ X₀);
    return (prob, unknowns(fol),1);
end


function createIVP2(fol, sys)
    e = full_equations(fol)
    d1 = defaults(fol)
    d2 = ModelingToolkit.missing_variable_defaults(fol) 
    if !isempty(d2)
        @warn "Missing initial values, $d2 is applied"
    end
    # Get default initial values
    d = merge(Dict(d1), Dict(d2))
    p = ModelingToolkit.parameters(fol)
    u = unknowns(fol)
    new_d = Dict(k => d[k] for k in p)

    filtered_d = Dict(kv for kv in new_d if !(kv.second isa IntervalArithmetic.Interval))
    new_u= Dict(kv for kv in new_d if (kv.second isa IntervalArithmetic.Interval))
    new_unk = keys(new_u)
    new_unk_vals = values(new_u)

    sys_unknowns = unknowns(sys)
    fol_unknowns = unknowns(fol)

    missing_in_sys = setdiff(fol_unknowns, sys_unknowns)
    s=symbolic_linear_solve(get_alg_eqs(fol), missing_in_sys)
    k = length(s)
    last_k_unknowns = fol_unknowns[end - k + 1 : end]
    nn = Dict(zip(last_k_unknowns, s))

    h=Symbolics.substitute(e,nn)
    for i in eachindex(h)
        h[i] = simplify(h[i])
    end
    eqs = h[1:end-k]
    eqs_n_1=Symbolics.fixpoint_sub(eqs,filtered_d)

    obs_map = Dict()
    for obs_eq in observed(fol)
        # Each obs_eq looks like "lhs ~ rhs".
        # We'll interpret it as "replace lhs with rhs".
        obs_map[obs_eq.lhs] = obs_eq.rhs
    end

    # Now substitute them into your eqs_n so that old variables
    # (like pend₊θ1(t)) become the new ones (like pid1₊θ(t)).
    eqs_n = Symbolics.fixpoint_sub(eqs_n_1, obs_map)


    first_unk=fol_unknowns[1:end-k]

    all_vars = [first_unk..., new_unk...]
    new_defaults = Dict{typeof(all_vars[1]), Any}()  # or Dict() if you don't mind the type
    for v in all_vars
        new_defaults[v] = d[v]
    end

    all_args = [all_vars...; ModelingToolkit.t]

    compiled_functions = Vector{Any}(undef, length(all_vars))
    resolved = Dict{Any,Any}()
    for eq in eqs_n
        resolved[eq.lhs] = eq.rhs
    end

    # For each variable in the full set, if it is in unresolved_vars, assign a zero derivative;
    # otherwise, try to build its derivative function from resolved_eqs.
    for (i, var) in enumerate(all_vars)
        if any(u -> isequal(u, var), new_unk)
            # For the newly promoted variable, its derivative should be zero.
            compiled_functions[i] = Symbolics.build_function(0, all_args...; expression=false)
        else
            # For the original unknowns, build the derivative function.
            key = Differential(t)(var)
            if haskey(resolved, key)
                rhs = resolved[key]
                compiled_functions[i] = Symbolics.build_function(rhs, all_args...; expression=false)
            else
                # If no differential equation is given, default to zero.
                compiled_functions[i] = Symbolics.build_function(0.0, all_args...; expression=false)
            end
        end
    end

    initial_values = [ new_defaults[v] for v in all_vars ]

    X₀ = IntervalBox(initial_values...);#, init_p...)

    function fol!(du, x, p, t)
        nvars = length(X₀);
        # Evaluate each derivative in order.
        for i in 1:nvars
            du[i] = compiled_functions[i](x[1:nvars]..., t) + zero(x[i]);
        end
    end

    order = length(X₀);
    prob = @ivp(x' = fol!(x), dim:order, x(0) ∈ X₀);
    return (prob,unknowns(fol),2)
end