using Ripserer, NPZ, JSON, Logging, OrderedCollections, CSV, DataFrames
using Base.Threads: @threads, nthreads

# DISULFIDE-AUGMENTED FULL PERSISTENT HOMOLOGY (Ripserer)
# 
# Produces the same three JSON outputs as persistent_homology.jl (PH_1,
# representatives, barcodes) 
#
# Ripserer forbids exact-zero off-diagonal edges, so every 0 off the diagonal is set to 10^-4
#settings (alg=:involuted, dim_max=1,
# no threshold) 
#
# Output three set of features :PH_1, representatives, barcodes
# 

const SCRIPT_DIR = @__DIR__
const DM_ROOT  = get(ENV, "AUG_DM_ROOT",
    normpath(joinpath(SCRIPT_DIR, "../results/distance_matrices/augmented")))
const OUT_ROOT = get(ENV, "PH_OUT_ROOT",
    normpath(joinpath(SCRIPT_DIR, "../disulfide_data/processed_data/persistent_homology")))
const STAPLE = 1e-4

const ID_MAP_PATH = get(ENV, "ID_MAP", "")

lp(p::AbstractString) = (Sys.iswindows() && !startswith(p, "\\\\?\\")) ? "\\\\?\\" * abspath(p) : p

function load_id_map(path)
    m = Dict{Tuple{String,String},String}()
    isempty(path) && return m
    isfile(path) || (@warn "ID_MAP not found: $path"; return m)
    df = CSV.read(path, DataFrame; types=String)
    for r in eachrow(df)
        m[(String(r.family), String(r.stem))] = String(r.uid)
    end
    @info "Loaded $(length(m)) id-map entries from $path"
    return m
end

function barcodes(PH, dim::Int)
    bars = collect.(collect(PH[dim + 1]))
    isempty(bars) && return Vector{Vector{Union{Float64,Nothing}}}()
    mat = Matrix(hcat(bars...)')
    jv(v) = isfinite(v) ? v : nothing
    return [Union{Float64,Nothing}[jv(mat[i, 1]), jv(mat[i, 2])] for i in 1:size(mat, 1)]
end

function representatives_of(PH, dim::Int)
    [
        [Int[v for v in collect(r.simplex)] for r in collect(c)]
        for c in representative.(PH[dim + 1])
    ]
end

function process_protein(npy_file, cls, dirs, idmap)::Bool
    stem = splitext(basename(npy_file))[1]
    pid  = get(idmap, (cls, stem), stem)   
    out_ph   = joinpath(dirs.ph,   "$pid.json")
    out_reps = joinpath(dirs.reps, "$pid.json")
    out_bars = joinpath(dirs.bars, "$pid.json")
    (isfile(lp(out_ph)) && isfile(lp(out_reps)) && isfile(lp(out_bars))) && return true
    try
        D = npzread(lp(npy_file))
        D = Matrix{Float64}(D)
        n = size(D, 1)
        @inbounds for i in 1:n, j in 1:n
            if i != j && D[i, j] == 0.0
                D[i, j] = STAPLE
            end
        end
        @inbounds for i in 1:n
            D[i, i] = 0.0
        end
        D = (D + D') / 2   

        PH = ripserer(D; dim_max=1, alg=:involuted)
        bar0  = barcodes(PH, 0)
        bar1  = barcodes(PH, 1)
        repre = representatives_of(PH, 1)

        open(lp(out_ph), "w") do io
            JSON.print(io, OrderedDict(
                :protein_id      => pid,
                :class           => cls,
                :dim_0_barcode   => bar0,
                :dim_1_barcode   => bar1,
                :representatives => repre,
            ), 2)
        end
        open(lp(out_reps), "w") do io
            JSON.print(io, OrderedDict(:representatives => repre), 2)
        end
        open(lp(out_bars), "w") do io
            JSON.print(io, OrderedDict(:dim_0 => bar0, :dim_1 => bar1), 2)
        end

        PH = D = bar0 = bar1 = repre = nothing
        GC.gc(false)
        return true
    catch e
        @error "✗ $pid ($cls)" exception=(e, catch_backtrace())
        return false
    end
end

function main()
    println("Disulfide-augmented FULL PH (Ripserer) | threads=$(nthreads())")
    println("  DM in : $DM_ROOT")
    println("  PH out: $OUT_ROOT")
    isdir(DM_ROOT) || error("missing $DM_ROOT")
    idmap = load_id_map(ID_MAP_PATH)
    classes = filter(d -> !startswith(d, ".") && isdir(joinpath(DM_ROOT, d)),
                     readdir(DM_ROOT))
    tot_ok = 0; tot = 0
    for cls in classes
        files = filter(f -> endswith(f, ".npy"),
                       readdir(joinpath(DM_ROOT, cls); join=true))
        base = joinpath(OUT_ROOT, cls)
        dirs = (ph=joinpath(base, "PH_1"),
                reps=joinpath(base, "representatives"),
                bars=joinpath(base, "barcodes"))
        mkpath(dirs.ph); mkpath(dirs.reps); mkpath(dirs.bars)
        ok = Threads.Atomic{Int}(0)
        t = @elapsed @threads for f in files
            process_protein(f, cls, dirs, idmap) && Threads.atomic_add!(ok, 1)
        end
        tot_ok += ok[]; tot += length(files)
        println("  $cls: $(ok[])/$(length(files)) in $(round(t; digits=1))s")
    end
    println("Done. $tot_ok/$tot proteins -> $OUT_ROOT")
end

main()
