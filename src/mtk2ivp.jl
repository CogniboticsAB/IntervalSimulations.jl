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


# These are the same "cleanup" tools from before.
# Make sure the pattern doesn't cause PCRE errors:
function remove_t_and_pluses(substring::AbstractString)
    no_t = replace(String(substring), r"\(t\)" => "")
    pat = r"((?:[0-9Ee.\+\-]+)?)(?:[A-Za-z0-9]+₊){1,}([A-Za-z0-9]+)"
    result = replace(no_t, pat => (match_sub::AbstractString) -> begin
        mm = match(pat, match_sub)
        mm === nothing && return match_sub
        return mm.captures[1] * mm.captures[2]
    end)
    return result
end

function clean_equation_lhs(lhs_str::AbstractString)
    pat = r"^Differential\(t\)\((.*)\)$"
    m = match(pat, lhs_str)
    if m !== nothing
        inside = m.captures[1]
        inside_cleaned = remove_t_and_pluses(inside)
        return "Differential(t)(" * inside_cleaned * ")"
    else
        return remove_t_and_pluses(lhs_str)
    end
end

function clean_equation_rhs(rhs_str::AbstractString)
    out = remove_t_and_pluses(rhs_str)

    # 1) Insert `*` after a digit if next char is '('
    out = replace(out, r"(?<=[0-9])(?=\()" => "*")

    # 2) Insert `*` after a digit if next char is letterlike
    #    (including Greek letters).  We use `(?<=[0-9])(?=[^\W\d_])`
    #    so that it excludes punctuation, digits, underscores, etc.
    out = replace(out, r"(?<=[0-9])(?=[^\W\d_])" => "*")

    # 3) Convert any remaining Unicode to ASCII approximations
    out = unidecode(out)
    return out
end

"""
    print_matlab_simplify_batch(eqs_n::Vector{Equation}, all_vars;
                                vpa_digits::Int=10,
                                print_vpa::Bool=true)

Given a list of final equations `eqs_n` and the `all_vars`
(e.g. `[phi(t), w(t), c, d]`) from Julia, this function:

1) Extracts the variable names automatically from `all_vars`.
2) Prints a single line of the form:
      syms phi w c d
   with any leftover Unicode replaced via unidecode.
3) For each equation eq_i, prints:
      eq1 = <cleaned RHS>;
      eq1_s = simplify(eq1);
   If `print_vpa=true`, it also prints:
      eq1_sf = vpa(eq1_s, 10);

The user can copy‐paste into MATLAB. By default, `vpa_digits=10`.
If you set `print_vpa=false`, it won't print the vpa lines.
"""
function print_matlab_simplify_batch(eqs_n::Vector{Equation}, all_vars,
                                     vpa_digits)
    # 1) Gather all variable names from all_vars, e.g. [phi(t), w(t), c, d].
    #    Then remove plus-chains, remove (t).
    varnames = String[]
    for v in all_vars
        raw = string(v)
        cleaned = remove_t_and_pluses(raw)
        # Also unidecode them in case they contain Greek letters
        cleaned = unidecode(cleaned)
        push!(varnames, cleaned)
    end

    # Make them unique & sorted for consistent output
    unique_vars = sort(unique(varnames))

    # 2) Print the syms line
    syms_line = "syms " * join(unique_vars, " ")
    println("\n# ------ MATLAB code for simplification ------")
    println(syms_line)

    # 3) For each eq in eqs_n, we create eq$i = <rhs>
    #    eq$i_s = simplify(eq$i)
    #    (optionally) eq$i_sf = vpa(eq$i_s, vpa_digits)
    for (i, eq) in enumerate(eqs_n)
        # Clean & unidecode the RHS
        rhs_str = clean_equation_rhs(string(eq.rhs))
        eqname = "eq$(i)"

        println(eqname, " = ", rhs_str, ";")
        println(eqname, "_s = simplify(", eqname, ");")
        println(eqname, "_sf = vpa(", eqname, "_s, ", vpa_digits, ");")
        
    end

    println("# ------ End of MATLAB snippet ------\n")
end

function createIVP2(fol, sys; vpa_digits = 10, print_matlab = false)
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

    obs_map = Dict()
    for obs_eq in observed(fol)
        # Each obs_eq looks like "lhs ~ rhs".
        # We'll interpret it as "replace lhs with rhs".
        obs_map[obs_eq.lhs] = obs_eq.rhs
    end

    missing_in_sys = setdiff(fol_unknowns, sys_unknowns)
    alg_eqs1 = get_alg_eqs(fol)
    l = length(alg_eqs1)
    alg_eqs=e[end-l+1:end]
    #alg_eqs=Symbolics.fixpoint_sub(alg_eqs1,obs_map)
    s=symbolic_linear_solve(alg_eqs, missing_in_sys)
    k = length(s)
    last_k_unknowns = fol_unknowns[end - k + 1 : end]

    new_var_obs = missing_in_sys .~ s

    nn = Dict(zip(last_k_unknowns, s))

    h=Symbolics.substitute(e,nn)
    h=h[1:end-k]
    for i in eachindex(h)
        h[i] = simplify(h[i])
    end
    eqs = h
    eqs_n=Symbolics.fixpoint_sub(eqs,filtered_d)

    for i in eachindex(h)
        eqs_n[i] = simplify(eqs_n[i])
    end

    #oos=Symbolics.substitute(oos,obs_map)
    # Now substitute them into your eqs_n so that old variables
    # (like pend₊θ1(t)) become the new ones (like pid1₊θ(t)).
    #eqs_n = Symbolics.fixpoint_sub(eqs_n_1, obs_map)


    first_unk=fol_unknowns[1:end-k]

    all_vars = [first_unk..., new_unk...]
    new_defaults = Dict{typeof(all_vars[1]), Any}()  # or Dict() if you don't mind the type
    for v in all_vars
        new_defaults[v] = d[v]
    end

    all_args = [all_vars...; ModelingToolkit.t]
    #println(all_args)
    compiled_functions = Vector{Any}(undef, length(all_vars))
    resolved = Dict{Any,Any}()


    # Print cleaned equations
    for eq in eqs_n
        resolved[eq.lhs] = eq.rhs
    end
    if print_matlab
        print_matlab_simplify_batch(eqs_n, all_vars, vpa_digits)
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
    #println(X₀)
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