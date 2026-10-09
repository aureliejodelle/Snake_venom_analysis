using Ripserer, DelimitedFiles, DataFrames, CSV, JSON, Logging
using Base.Threads: @threads, nthreads
using OrderedCollections

# PATHS: read from environment variables set by run_pipeline.py
const SCRIPT_DIR  = @__DIR__
const INPUT_ROOT  = get(ENV, "PH_INPUT_ROOT",
    normpath(joinpath(SCRIPT_DIR, "../Database/raw_data/coordinates_data")))
const OUTPUT_ROOT = get(ENV, "PH_OUTPUT_ROOT",
    normpath(joinpath(SCRIPT_DIR, "../Database/filtered_processed_data/Persistent_homology")))

# Sub-folder names
const DIR_PH       = "PH_1"
const DIR_REPS     = "representatives"
const DIR_BARCODES = "barcodes"


"""
Convert PH barcodes for a given dimension into a plain
Vector{Vector{Union{Float64,Nothing}}}.
"""
function barcodes(PH, dim::Int)
    bars = collect.(collect(PH[dim+1]))
    isempty(bars) && return Vector{Vector{Union{Float64,Nothing}}}()
    mat = Matrix(hcat(bars...)')
    to_json_val(v::Float64) = isfinite(v) ? v : nothing
    return [
        Union{Float64,Nothing}[to_json_val(mat[i,1]), to_json_val(mat[i,2])]
        for i in 1:size(mat,1)
    ]
end

"""
Extract simplex representatives as plain Vector{Vector{Vector{Int}}}.
"""
function representatives(PH, dim::Int)
    [
        [Int[v for v in collect(r.simplex)] for r in collect(c)]
        for c in representative.(PH[dim+1])
    ]
end

"""
Create all output subdirectories for a protein class.
"""
function make_output_dirs(class_name::String)
    base = joinpath(OUTPUT_ROOT, class_name)
    dirs = (
        ph       = joinpath(base, DIR_PH),
        reps     = joinpath(base, DIR_REPS),
        barcodes = joinpath(base, DIR_BARCODES),
    )
    mkpath.(values(dirs))
    return dirs
end


"""
Scan INPUT ROOT and return all subdirectory names which are the different classes.
"""
function detect_classes()::Vector{String}
    if !isdir(INPUT_ROOT)
        error("Input root not found: $INPUT_ROOT")
    end
    all_dirs = filter(d -> isdir(joinpath(INPUT_ROOT, d)), readdir(INPUT_ROOT))
    # Exclude non relevant folders
    classes = filter(d -> !startswith(d, "."), all_dirs)
    isempty(classes) && @warn "No subdirectories found in $INPUT_ROOT"
    @info "Detected $(length(classes)) class(es): $(join(classes, ", "))"
    return classes
end


"""
Compute persistent homology for one protein and save JSON outputs.
"""
function process_protein(input_file::String, class_name::String, dirs)::Bool
    protein_id = splitext(basename(input_file))[1]

    #Skip if all three outputs already exist
    out_ph   = joinpath(dirs.ph,       "$protein_id.json")
    out_reps = joinpath(dirs.reps,     "$protein_id.json")
    out_bars = joinpath(dirs.barcodes, "$protein_id.json")
    if isfile(out_ph) && isfile(out_reps) && isfile(out_bars)
        @info "⏭  Skipping $protein_id ($class_name) — outputs already exist"
        return true   # count as success so pipeline stats stay accurate
    end

    try
        # Load Cα coordinates
        grid = Matrix{Float64}(CSV.read(input_file, DataFrame)) |> eachrow .|> Tuple

        # Compute PH1 reps=true gives minimal representatives via column reduction
        PH = ripserer(grid; dim_max=1, alg=:involuted)

        # Extract serializable data
        bar0  = barcodes(PH, 0)
        bar1  = barcodes(PH, 1)
        repre = representatives(PH, 1)

        # PH1: full data (barcodes + representatives)
        open(joinpath(dirs.ph, "$protein_id.json"), "w") do io
            JSON.print(io, OrderedDict(
                :protein_id      => protein_id,
                :class           => class_name,
                :dim_0_barcode   => bar0,
                :dim_1_barcode   => bar1,
                :representatives => repre,
            ), 2)
        end

        # representatives: cycles 
        open(joinpath(dirs.reps, "$protein_id.json"), "w") do io
            JSON.print(io, OrderedDict(:representatives => repre), 2)
        end

        # barcodes: birth/death pairs 
        open(joinpath(dirs.barcodes, "$protein_id.json"), "w") do io
            JSON.print(io, OrderedDict(:dim_0 => bar0, :dim_1 => bar1), 2)
        end

        @info "✓ $protein_id ($class_name)"

        PH = grid = bar0 = bar1 = repre = nothing
        GC.gc(true)
        return true

    catch e
        @error "✗ Failed: $protein_id ($class_name)" exception=(e, catch_backtrace())
        GC.gc(true)
        return false
    end
end



function process_class(class_name::String)
    input_dir = joinpath(INPUT_ROOT, class_name)
    csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

    if isempty(csv_files)
        @warn "No CSV files in $class_name — skipping"
        return (total=0, success=0, failed=0, skipped=0)
    end

    dirs = make_output_dirs(class_name)

    # Count how many are already done before spawning threads
    already_done = count(csv_files) do file
        pid = splitext(basename(file))[1]
        isfile(joinpath(dirs.ph,       "$pid.json")) &&
        isfile(joinpath(dirs.reps,     "$pid.json")) &&
        isfile(joinpath(dirs.barcodes, "$pid.json"))
    end
    to_run = length(csv_files) - already_done
    @info "Processing '$class_name' — $(length(csv_files)) total | $already_done already done | $to_run to run | $(nthreads()) thread(s)"

    success_count = Threads.Atomic{Int}(0)
    fail_count    = Threads.Atomic{Int}(0)

    # Fully parallel
    @threads for file in csv_files
        ok = process_protein(file, class_name, dirs)
        Threads.atomic_add!(ok ? success_count : fail_count, 1)
    end

    return (total=length(csv_files), success=success_count[], failed=fail_count[], skipped=already_done)
end


# MAIN
function main()
    println("=" ^ 60)
    println("PERSISTENT HOMOLOGY COMPUTATION")
    println("=" ^ 60)
    println("Input  : $INPUT_ROOT")
    println("Output : $OUTPUT_ROOT")
    println("Threads: $(nthreads())")
    println()

    classes = detect_classes()
    overall = (total=0, success=0, failed=0, skipped=0)

    for class_name in classes
        println("\n" * "-" ^ 60)
        println("CLASS: $class_name")
        println("-" ^ 60)

        t     = @elapsed stats = process_class(class_name)
        overall = (
            total   = overall.total   + stats.total,
            success = overall.success + stats.success,
            failed  = overall.failed  + stats.failed,
            skipped = overall.skipped + stats.skipped,
        )
        println("  Done in $(round(t; digits=1))s | " *
                "success=$(stats.success)  skipped=$(stats.skipped)  " *
                "failed=$(stats.failed)  total=$(stats.total)")
    end

    println("\n" * "=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    println("  Classes processed : $(length(classes))")
    println("  Total proteins    : $(overall.total)")
    println("  Success           : $(overall.success)")
    println("  Skipped           : $(overall.skipped)  (outputs already existed)")
    println("  Failed            : $(overall.failed)")
    println("=" ^ 60)
end

main()