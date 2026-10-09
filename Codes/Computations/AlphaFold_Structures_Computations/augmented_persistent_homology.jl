using Ripserer, DataFrames, CSV, JSON, Logging, Distances, OrderedCollections
using Base.Threads: @threads, nthreads

# C-alpha-LEVEL DISULFIDE-AUGMENTED PERSISTENT HOMOLOGY (Ripserer)
# For every protein, Ripserer runs TWICE on the C-alpha distance matrix:
#   standard  : plain Euclidean  C-alpha distances
#   augmented : each disulfide staple ( C-alpha_i– C-alpha_j of two bonded cysteines
#
# Output: h1 bars

const SCRIPT_DIR   = @__DIR__
# dataset
const INPUT_ROOT   = get(ENV, "CA_INPUT_ROOT", normpath(joinpath(SCRIPT_DIR, "../ca_clouds")))
const SSBONDS_ROOT = get(ENV, "SS_ROOT",       normpath(joinpath(SCRIPT_DIR, "../ss_bonds_ca")))
const OUTPUT_ROOT  = get(ENV, "H1_OUTPUT_ROOT", normpath(joinpath(SCRIPT_DIR, "../h1_bars_ca")))
const STAPLE       = 1e-4
const THRESH       = 12.0   # max filtration

# Fix long protein filenames.
lp(p::AbstractString) = (Sys.iswindows() && !startswith(p, "\\\\?\\")) ? "\\\\?\\" * abspath(p) : p

function h1_bars(PH)
    bars = collect.(collect(PH[2]))
    isempty(bars) && return Vector{Vector{Union{Float64,Nothing}}}()
    mat = Matrix(hcat(bars...)')
    jv(v) = isfinite(v) ? v : nothing
    return [Union{Float64,Nothing}[jv(mat[i,1]), jv(mat[i,2])] for i in 1:size(mat,1)]
end

function load_ca_pairs(cls, pid)
    f = joinpath(SSBONDS_ROOT, cls, "$pid.json")
    isfile(lp(f)) || return Tuple{Int,Int}[]
    d = JSON.parsefile(lp(f))
    return [(Int(p[1]), Int(p[2])) for p in get(d, "ca_pairs", [])]
end

function process_protein(input_file, cls, out_std_dir, out_aug_dir)::Bool
    pid = splitext(basename(input_file))[1]
    out_std = joinpath(out_std_dir, "$pid.json")
    out_aug = joinpath(out_aug_dir, "$pid.json")
    (isfile(lp(out_std)) && isfile(lp(out_aug))) && return true
    try
        pts = Matrix{Float64}(CSV.read(lp(input_file), DataFrame))
        D = pairwise(Euclidean(), pts; dims=1)

        PH_s = ripserer(D; dim_max=1, threshold=THRESH)
        open(lp(out_std), "w") do io; JSON.print(io, OrderedDict(:dim_1 => h1_bars(PH_s))); end

        Da = copy(D)
        for (i, j) in load_ca_pairs(cls, pid)
            if 1 <= i+1 <= size(Da,1) && 1 <= j+1 <= size(Da,1)
                Da[i+1, j+1] = STAPLE; Da[j+1, i+1] = STAPLE
            end
        end
        PH_a = ripserer(Da; dim_max=1, threshold=THRESH)
        open(lp(out_aug), "w") do io; JSON.print(io, OrderedDict(:dim_1 => h1_bars(PH_a))); end
        return true
    catch e
        @error "✗ $pid ($cls)" exception=(e, catch_backtrace())
        return false
    end
end

function main()
    println(" C-alpha disulfide-augmented PH (Ripserer) | threads=$(nthreads())")
    isdir(INPUT_ROOT) || error("missing $INPUT_ROOT (run make_ca_clouds.py first)")
    classes = filter(d -> !startswith(d, ".") && isdir(joinpath(INPUT_ROOT, d)),
                     readdir(INPUT_ROOT))
    tot_ok = 0
    for cls in classes
        files = filter(f -> endswith(f, ".csv"), readdir(joinpath(INPUT_ROOT, cls); join=true))
        out_std_dir = joinpath(OUTPUT_ROOT, "standard",  cls); mkpath(out_std_dir)
        out_aug_dir = joinpath(OUTPUT_ROOT, "augmented", cls); mkpath(out_aug_dir)
        ok = Threads.Atomic{Int}(0)
        t = @elapsed @threads for f in files
            process_protein(f, cls, out_std_dir, out_aug_dir) && Threads.atomic_add!(ok, 1)
        end
        tot_ok += ok[]
        println("  $cls: $(ok[])/$(length(files)) in $(round(t;digits=1))s")
    end
    println("Done. $tot_ok proteins -> $OUTPUT_ROOT")
end

main()
