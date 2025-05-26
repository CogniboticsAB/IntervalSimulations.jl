# MTKPreparation.jl

"""
remove_t_and_pluses(substring::AbstractString)

Removes (t) from variables names and attempts to delete chained subscript-style variables like `x₊y₊z` with only the final symbol (`z`)
Called by print_matlab_simplify_batch.
"""
function remove_t_and_pluses(substring::AbstractString)
    no_t = Base.replace(String(substring), r"\(t\)" => "")
    pat = r"((?:[0-9Ee.\+\-]+)?)(?:[A-Za-z0-9]+₊){1,}([A-Za-z0-9]+)"
    result = Base.replace(no_t, pat => (match_sub::AbstractString) -> begin
        mm = match(pat, match_sub)
        mm === nothing && return match_sub
        return mm.captures[1] * mm.captures[2]
    end)
    return result
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
function print_matlab_simplify_batch(eqs_n::Vector{MTK.Equation}, all_vars,
                                     vpa_digits)
    # 1) Gather all variable names from all_vars, e.g. [phi(t), w(t), c, d].
    #    Then remove plus-chains, remove (t).
    varnames = String[]
    for v in all_vars
        raw = string(v)
        cleaned = remove_t_and_pluses(raw)
        # Also unidecode them in case they contain Greek letters
        cleaned = Unidecode.unidecode(cleaned)
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

"""
    clean_equation_rhs(rhs_str::AbstractString)
    Attempts to clean up the symbolic expressions from MTK. Adding multiplication signs where needed and converts unicode characters using the Unidecode package.
"""
function clean_equation_rhs(rhs_str::AbstractString)
    out = remove_t_and_pluses(rhs_str)

    # 1) Insert `*` after a digit if next char is '('
    out = Base.replace(out, r"(?<=[0-9])(?=\()" => "*")

    # 2) Insert `*` after a digit if next char is letterlike
    #    (including Greek letters).  We use `(?<=[0-9])(?=[^\W\d_])`
    #    so that it excludes punctuation, digits, underscores, etc.
    out = Base.replace(out, r"(?<=[0-9])(?=[^\W\d_])" => "*")

    # 3) Convert any remaining Unicode to ASCII approximations
    out = Unidecode.unidecode(out)
    return out
end


"""
    createIVP(sys)

Prepares an MTK system for validated reachability analysis. Automatically simplifies the system,
resolves initial values (including intervals), extracts algebraic equations if needed,
and returns a reachability-ready IVP.
"""
function createIVP(sys; vpa_digits = 10, print_matlab = false)
    fol = ModelingToolkit.structural_simplify(sys, simplify =true)

    e = ModelingToolkit.full_equations(fol)
    d1 = ModelingToolkit.defaults(fol)
    d2 = ModelingToolkit.missing_variable_defaults(fol) 
    if !isempty(d2)
        @warn "Missing initial values, $d2 is applied"
    end

    d = merge(Dict(d1), Dict(d2))
    p = ModelingToolkit.parameters(fol)
    u = ModelingToolkit.unknowns(fol)
    new_d = Dict(k => d[k] for k in p)

    filtered_d = Dict(kv for kv in new_d if !(kv.second isa IntervalArithmetic.Interval))
    new_u = Dict(kv for kv in new_d if kv.second isa IntervalArithmetic.Interval)
    new_unk = keys(new_u)
    new_unk_vals = values(new_u)

    fol_unknowns = ModelingToolkit.unknowns(fol)
    obs_map = Dict()
    for obs_eq in ModelingToolkit.observed(fol)
        obs_map[obs_eq.lhs] = obs_eq.rhs
    end

    missing_in_sys = setdiff(fol_unknowns, ModelingToolkit.unknowns(sys))
    alg_eqs1 = ModelingToolkit.get_alg_eqs(fol)
    l = length(alg_eqs1)
    alg_eqs = e[end-l+1:end]
    s = Symbolics.symbolic_linear_solve(alg_eqs, missing_in_sys)
    k = length(s)
    last_k_unknowns = fol_unknowns[end - k + 1 : end]

    new_var_obs = missing_in_sys .~ s
    nn = Dict(zip(last_k_unknowns, s))
    h = Symbolics.substitute(e, nn)
    h = h[1:end-k]
    for i in eachindex(h)
        h[i] = Symbolics.simplify(h[i])
    end
    eqs = h
    eqs_n = Symbolics.fixpoint_sub(eqs, filtered_d)

    for i in eachindex(h)
        eqs_n[i] = Symbolics.simplify(eqs_n[i])
    end

    first_unk = fol_unknowns[1:end-k]
    all_vars = [first_unk...; new_unk...]
    new_defaults = Dict(v => d[v] for v in all_vars)

    all_args = [all_vars...; ModelingToolkit.t]
    compiled_functions = Vector{Any}(undef, length(all_vars))
    resolved = Dict(eq.lhs => eq.rhs for eq in eqs_n)

    if print_matlab
        print_matlab_simplify_batch(eqs_n, all_vars, vpa_digits)
    end


    for (i, var) in enumerate(all_vars)
        if any(u -> isequal(u, var), new_unk)
            compiled_functions[i] = Symbolics.build_function(0, all_args...; expression=false)
        else
            key = Symbolics.Differential(ModelingToolkit.t)(var)
            compiled_functions[i] = Symbolics.build_function(get(resolved, key, 0.0), all_args...; expression=false)
        end
    end

    initial_values = [new_defaults[v] for v in all_vars]
    X₀ = IntervalArithmetic.IntervalBox(initial_values...)

    function fol!(du, x, p, t)
        nvars = length(X₀)
        for i in 1:nvars
            du[i] = compiled_functions[i](x[1:nvars]..., t) + zero(x[i])
        end
    end

    order = length(X₀)
    prob = @ivp(x' = fol!(x), dim = order, x(0) ∈ X₀)
    return (prob, ModelingToolkit.unknowns(fol), 2)
end
