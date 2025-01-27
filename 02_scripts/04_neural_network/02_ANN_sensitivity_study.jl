using JLD2
using BSON
using Dates 
using Interpolations
using Flux
using Random
using Statistics
using Distributions
using Plots
using ProgressMeter
using Printf
## new
using FileIO
#using CairoMakie
using LinearAlgebra
using Trapz

#sample_versions = ["geometrical_samples\\v6"]
#sample_versions = ["mechanical_samples\\v4finer"]
#sample_versions = ["selected_param_samples2\\v1", "selected_param_samples2\\v2"]
sample_versions = ["new_mechanical_samples\\v1"]

total_epochs = 10000
learning_rate = 0.001
random_seed = 1234
Random.seed!(random_seed) #set random seed for reproducibility (hyperparameter changing)


# Define parameter names based on the sample type (for PLOTS)
mech_params_names = ["GI", "GII", "tI", "tII", "i_fric", "b_fric"]
#mech_params_names = ["GI", "GII", "t", "i_fric", "b_fric"]
geom_params_names = ["fd", "D", "L/D", "θ", "ell/r", "φ", "bl_d"] # θ is contact angle; φ is fiber rotation
#selected_params01_names = ["GII", "t", "fd", "D", "bl_d"]
selected_params02_names = ["GII", "t", "fd", "ell/r", "bl_d"]

### ------ Filestructure and Dir creation ------ ###
root_results_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\03_results" #results folder

common_prefix = dirname(sample_versions[1])
version_names = map(basename, sample_versions)
combined_name = joinpath(common_prefix,join(version_names, "_"))
results_dir = joinpath(root_results_dir, combined_name, "01_filtered_data")
results_dir_ANN = joinpath(root_results_dir, combined_name, "02_ANN")

if !isdir(results_dir_ANN)
    mkpath(results_dir_ANN)
    println("Created results directory at: $results_dir_ANN")
end

# Function to create a new run directory
function create_new_run_dir(base_dir::String)
    # Find existing und build run directory
    existing_runs = filter(name -> occursin(r"^run\d{2}$", name), readdir(base_dir))
    run_numbers = parse.(Int, [match(r"^run(\d{2})$", name).captures[1] for name in existing_runs])
    next_run_number = isempty(run_numbers) ? 1 : maximum(run_numbers) + 1
    run_dir_name = @sprintf("run%02d", next_run_number)
    run_dir = joinpath(base_dir, run_dir_name)
    # Create the directory
    mkpath(run_dir)
    println("Created new run directory at: $run_dir")
    return run_dir
end

results_dir_ANN_run = create_new_run_dir(results_dir_ANN)
### ------------------------------------------- ###


### --- Load filtered data -------------------- ### 

# Function to load Xs, Ys, and clean_params from a .jld2 file
function load_data_from_jld2(filename)
    @load filename Xs Ys clean_params indices_clean
    println("Data loaded from $filename")
    return Xs, Ys, clean_params, indices_clean
end

# Load data if there are more directories/versions specified 
Ys_combined = []
clean_params_combined = []
indices_combined = []
Xs = nothing  # This ensures Xs is defined before the loop
for sample in sample_versions
    load_data_file = joinpath(root_results_dir, sample, "01_filtered_data", "clean_data.jld2")
    Xs_temp, Ys, clean_params, indices = load_data_from_jld2(load_data_file)
    if Xs === nothing
        global Xs = Xs_temp
    end
    # Append Ys and clean_params, and indices
    append!(Ys_combined, Ys)
    append!(clean_params_combined, clean_params)
    append!(indices_combined, indices)  # Collect the original indices
end
### -------------------------------------------- ### 


#######################
###### ANN Model ######
#######################

# Normalize input parameters
parameter_ranges_from_data = extrema.(eachrow(reduce(hcat, clean_params_combined)))
normalized_params = map(clean_params_combined) do p
    [(p[i]-l)/(u-l) for (i,(l,u)) in enumerate(parameter_ranges_from_data)]
end

# Select training and test data
N_training = Int(ceil(length(Ys_combined)*4/5))

# Randomize IDs for trying different sets
training_ids = Set(shuffle(1:length(Ys_combined))[1:N_training])
test_ids = setdiff(Set(1:length(Ys_combined)),training_ids)
# Compose training and test sets (input, label)
data_training = [ (normalized_params[id], Ys_combined[id]) for id in training_ids]
original_ids_training = [ indices_combined[id] for id in training_ids]  # Store original IDs for training data
clean_params_combined_training = [ clean_params_combined[id] for id in training_ids]

data_test = [ (normalized_params[id], Ys_combined[id]) for id in test_ids]
original_ids_test = [ indices_combined[id] for id in test_ids]  # Store original IDs for test data
clean_params_combined_test = [ clean_params_combined[id] for id in test_ids]

### --- This is the big model --- ###

# Determine size of layers
N_inp = length(first(clean_params_combined))
N_out = length(Xs)

# This is the actual model
model = Chain(
    Dense(N_inp => 4*N_inp, celu),
    Dropout(0.4), # to reduce overfit
    Dense(4*N_inp => 4*N_inp, celu),
    Dropout(0.4),
    Dense(4*N_inp => N_out)
) |> f64

# This is the optimizer
optim = Flux.setup(Flux.Adam(learning_rate), model)

# This is the batch loader with the minibatch size
batch_size = 8
batches = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true) #batchsize=2 is worse for mechsamples...

function train_model(total_epochs)
    # Initialize variables
    start_time = Dates.now()
    losses_training = Float64[]
    losses_validation = Float64[]
    best_validation_loss = Inf
    best_epoch = 0
    best_model = Flux.state(model)

    # Initialize progress bar without showvalues
    p = Progress(total_epochs, desc="Training Progress")

    for epoch in 1:total_epochs
        # Loop over minibatches
        for batch in batches
            val, grads = Flux.withgradient(model) do m
                mean(Flux.Losses.mae(m(input), label) for (input, label) in batch)
            end
            Flux.update!(optim, model, grads[1])
        end

        # Compute current training and validation losses
        current_training_loss = mean(Flux.Losses.mae(model(x), y) for (x, y) in data_training)
        current_validation_loss = mean(Flux.Losses.mae(model(x), y) for (x, y) in data_test)

        # Update progress bar with current values
        ProgressMeter.update!(p, epoch; showvalues = [(:Train_Loss, round(current_training_loss, digits=6)),
                                                        (:Val_Loss, round(current_validation_loss, digits=6))])

        # Store losses
        push!(losses_training, current_training_loss)
        push!(losses_validation, current_validation_loss)

        # Update best model if validation loss improves
        if current_validation_loss < best_validation_loss
            best_validation_loss = current_validation_loss
            best_epoch = epoch
            best_model = Flux.state(model)
        end
    end

    finish!(p)
    Flux.loadmodel!(model, best_model)

    println("\nTraining complete.")
    println("Best validation loss at epoch $best_epoch with loss $best_validation_loss")

    total_runtime = Dates.now() - start_time
    return best_model, losses_training, losses_validation, best_epoch, best_validation_loss, total_runtime
end

best_model, losses_training, losses_validation, best_epoch, best_validation_loss, total_runtime = train_model(total_epochs)
best_training_loss = losses_training[best_epoch]

# Save the model after training
model_file = joinpath(results_dir_ANN_run, "trained_model.bson")
BSON.@save model_file model
println("Trained model saved to $model_file")

##############################
### PLOT ANN Model results ###
##############################

############# Get parameter names for Plotting #############
# Decide which parameter names to use based on the common prefix (geometrical_sampls, mechanical_samples, etc.)
if occursin("mechanical_samples", common_prefix)
    param_names = mech_params_names
elseif occursin("geometrical_samples", common_prefix)
    param_names = geom_params_names
elseif occursin("selected_param_samples2", common_prefix)
    param_names = selected_params02_names
else
    param_names = vcat(mech_params_names, geom_params_names)
end

parameter_ranges = extrema.(eachrow(reduce(hcat, clean_params_combined)))
range_text = "Parameter ranges:\n " * join([ "$(param_names[i]): (" * string(round(l, digits=4)) * "-" * string(round(u, digits=4)) * ")" 
                                           for (i, (l, u)) in enumerate(parameter_ranges)], ", ")

function generate_param_string(sample_idx, dataset_type::Symbol)
    # Choose the dataset based on `dataset_type`
    params = dataset_type == :test ? clean_params_combined_test[sample_idx] : clean_params_combined_training[sample_idx]
    
    # Generate the parameter string
    param_str = join([string(param_names[i]) * ": " * string(round(p, digits=4)) for (i, p) in enumerate(params)], ", ")
    return param_str
end    
#######                                      

save_plot_loss_log = joinpath(results_dir_ANN_run, "loss_f_log.png")
save_plot_worst = joinpath(results_dir_ANN_run, "n_worst.png")
save_plot_best = joinpath(results_dir_ANN_run, "n_best.png")
save_plot_median = joinpath(results_dir_ANN_run, "n_median.png")
save_gradient = joinpath(results_dir_ANN_run, "gradient.png")

# Plotting the loss
p_loss_log = plot([losses_training, losses_validation], label=["training loss" "validation loss"], yscale=:log10, xlabel="epochs", ylabel="MSE", size=(1200, 900))
vline!(p_loss_log, [best_epoch], label="Best Epoch", linestyle=:dash, color=:red)
savefig(p_loss_log, save_plot_loss_log)
savefig(p_loss_log, replace(save_plot_loss_log, ".png" => ".svg"))  # Additional SVG save
display(p_loss_log)

# Plotting the N worst approximations
test_losses = [Flux.Losses.mae(model(d[1]),d[2]) for d in data_test]
n_max = sort(collect(1:length(test_losses)), by=i->test_losses[i], rev=true)
p_worst = plot(layout=grid(2,3), plot_title=range_text, plot_titlefontsize=6,
    [plot(Xs, [model(data_test[idx][1]), data_test[idx][2]], 
    labels=["prediction" "truth"], 
    title="WORST SAMPLE [original ID: $(original_ids_test[idx])]\n$(generate_param_string(idx, :test))", size=(1200, 900),
        titlefont=5, ylims=(0, 20), xticks=0:0.02:0.1, yticks=0:2:20) for idx in n_max[1:6]]...)
savefig(p_worst, save_plot_worst)
savefig(p_worst, replace(save_plot_worst, ".png" => ".svg"))  # Save as SVG
display(p_worst)

# Plotting the N best approximations
n_min = sort(collect(1:length(test_losses)), by=i->test_losses[i])  # Sort in ascending order for the best losses
p_best = plot(layout=grid(2, 3), plot_title=range_text, plot_titlefontsize=6,
    [plot(Xs, [model(data_test[idx][1]), data_test[idx][2]], 
    labels=["prediction" "truth"], 
    title="BEST SAMPLE [original ID: $(original_ids_test[idx])]\n$(generate_param_string(idx, :test))", size=(1200, 900),
        titlefont=5, ylims=(0, 20), xticks=0:0.02:0.1, yticks=0:2:20) for idx in n_min[1:6]]...)
savefig(p_best, save_plot_best)
savefig(p_best, replace(save_plot_best, ".png" => ".svg"))  # Save as SVG
display(p_best)

# Plotting samples close to the median of all samples 
middle_start = Int(floor(length(test_losses) / 2)) - 3  # Start 3 positions before the middle
middle_end = middle_start + 5  # Take 6 samples
n_middle = n_max[middle_start:middle_end]
p_median = plot(layout=grid(2, 3), plot_title=range_text, plot_titlefontsize=6,
    [plot(Xs, [model(data_test[idx][1]), data_test[idx][2]], 
    labels=["prediction" "truth"], 
    title="MEDIAN SAMPLE [original ID: $(original_ids_test[idx])]\n$(generate_param_string(idx, :test))", size=(1200, 900),
        titlefont=5, ylims=(0, 20), xticks=0:0.02:0.1, yticks=0:2:20) for idx in n_middle]...)
savefig(p_median, save_plot_median)
savefig(p_median, replace(save_plot_median, ".png" => ".svg"))  # Save as SVG
display(p_median)


# Compute the range of function values
base = [model(collect(ps)) for ps in Iterators.product([range(0,1, length=3) for _ in 1:N_inp]...)]
# Compute the derivative for all parameters at all locations in base
sens = [Flux.jacobian(model, collect(ps))[1] for ps in Iterators.product([range(0,1, length=3) for _ in 1:N_inp]...)]
# Fit normal distributions to the gradient data
dists = [[fit(Normal,[sens[s][i,pidx] for s in eachindex(sens)]) for i in 1:N_out] for pidx in 1:N_inp]

# Plot the gradient data
colors = palette(:rainbow, N_inp)
plt2 = plot(ylabel="Gradient", xticks=0:0.01:0.1)
for (idx,v) in enumerate(param_names)
    # Compute quartile
    quant_map = hcat(quantile.(dists[idx],Ref([0.25, 0.75]))...)
    # Compute mean
    μs = map(x->x.μ, dists[idx])
    # Plot inner quartile range as shaded area
    plot!(plt2, Xs, μs, color=colors[idx], lw=0, ribbon=(μs - quant_map[1,:], quant_map[end,:]-μs), label=:none)
    # Plot mean
    plot!(plt2, Xs, μs, color=colors[idx], label=v)
end

# Compute visualization of base values
_base = reduce(hcat,base)
qtly = quantile.(eachrow(_base), Ref([0.01, 0.05, 0.25, 0.75, 0.95, 0.99]))
#_base_01 = (x -> x[1]).(qtly)  # 1% quantile
_base_05 = (x -> x[2]).(qtly)  # 5% quantile
_base_25 = (x -> x[3]).(qtly)  # 25% quantile
_base_75 = (x -> x[4]).(qtly)  # 75% quantile
_base_95 = (x -> x[5]).(qtly)  # 95% quantile
#_base_99 = (x -> x[6]).(qtly)  # 99% quantile

_base_mean = mean.(eachrow(_base))

# Plot inner quartile range of base values
colors_red = palette(:reds)
plt1 = plot(Xs, _base_25, xticks=0:0.01:0.1, yticks=0:2:12, lw=0, fillrange=_base_75, alpha=0.5, color=colors_red[1], label=:none, ylabel="Function")
plot!(plt1, Xs, _base_mean, color=colors_red[2], label="Mean")
plot!(plt1, Xs, _base_25, lw=2, color=:black, linestyle=:dash, label="25% Quantile")
plot!(plt1, Xs, _base_75, lw=2, color=:black, linestyle=:dash, label="75% Quantile")

plot!(plt1, Xs, _base_05, lw=2, color=:gray, linestyle=:dot, label="5% Quantile")
plot!(Xs, _base_05, lw=0, fillrange=_base_25, alpha=0.10, color=:black, label=:none)
plot!(plt1, Xs, _base_95, lw=2, color=:gray, linestyle=:dot, label="95% Quantile")
plot!(Xs, _base_75, lw=0, fillrange=_base_95, alpha=0.10, color=:black, label=:none)
#plot!(plt1, Xs, _base_01, lw=2, color=:gray, linestyle=:dashdot, label="1% Quantile")
#plot!(plt1, Xs, _base_99, lw=2, color=:gray, linestyle=:dashdot, label="99% Quantile")



# Combine both plots and display
p_gradient = plot(layout=grid(2,1), plt1, plt2, size=(1200, 900))
savefig(p_gradient, save_gradient)
savefig(p_gradient, replace(save_gradient, ".png" => ".svg"))
display(p_gradient)


#### NEW: PLOT ALL VALIDATION SAMPLES TO COMPARE

# Determine number of rows and columns for layout
n_cols = 4  # or any manageable number
n_rows = ceil(Int, length(n_max) / n_cols)

# Create subplot layout for all samples

p_all_samples = plot(layout=grid(n_rows,n_cols), plot_title=range_text, plot_titlefontsize=6,
    [plot(Xs, [model(data_test[idx][1]), data_test[idx][2]], 
    labels=["test predict" "fea truth"], 
    title="SAMPLE [original ID: $(original_ids_test[idx])]\n$(generate_param_string(idx, :test))", size=(n_cols*400, n_rows*400),
        titlefont=5, ylims=(0, 20)) for idx in n_max]...)

# Save and display
savefig(joinpath(results_dir_ANN_run, "all_validation.png"))
display(p_all_samples)

#################################

#### NEW: PLOT ALL TRAINING SAMPLES TO COMPARE

# Compute losses for each training sample
train_losses = [Flux.Losses.mae(model(data[1]), data[2]) for data in data_training]

# Sort indices based on training losses (ascending for best samples)
sorted_train_indices = sort(collect(1:length(train_losses)), by=i->train_losses[i], rev=true) #starting from worst

# Determine number of rows and columns for layout for sorted training samples
n_cols = 4
n_rows = ceil(Int, length(sorted_train_indices) / n_cols)

# Create subplot layout for sorted training samples
p_all_training_samples = plot(layout=grid(n_rows, n_cols), plot_title=range_text,
    plot_titlefontsize=6, size=(n_cols*400, n_rows*400),
    [plot(Xs, [model(data_training[idx][1]), data_training[idx][2]], 
    labels=["training predict" "fea truth"], 
    title="SAMPLE [original ID: $(original_ids_training[idx])]\n$(generate_param_string(idx, :training))", 
    titlefont=5, ylims=(0, 20)) for idx in sorted_train_indices]...)

# Save and display
savefig(joinpath(results_dir_ANN_run, "all_training.png"))
display(p_all_training_samples)





#######################################
##### LOG FILE for ANN parameters #####
#######################################

# Extract optimizer info
optimizer_name = split(typeof(optim.layers[1].weight).parameters[1] |> string, ".")[end]
optimizer_info = string(optim.layers[1].weight)
optimizer_details = optimizer_details = match(r"Leaf\([^)]*\)", optimizer_info).match

# Calculate the number of training and test samples
training_count = length(training_ids)
test_count = length(test_ids)

function format_runtime(runtime::Dates.Period)
    # Convert the period to total seconds
    total_seconds = Dates.value(runtime) / 1000  # Since Dates.value returns milliseconds
    hours = floor(Int, total_seconds / 3600)
    minutes = floor(Int, (total_seconds % 3600) / 60)
    seconds = round(total_seconds % 60, digits=2)
    return "$(hours)h $(minutes)m $(seconds)s"
end


function log_parameters(filepath::String; kwargs...)
    open(filepath, "w") do io
        println(io, "Model and Training Parameters:")
        for (key, value) in kwargs
            if key == :Total_Runtime
                formatted_runtime = format_runtime(value)
                println(io, "$key: $formatted_runtime")
            else
                println(io, "$key: $value")
            end
        end
    end
    println("Parameters logged to $filepath")
end

using OrderedCollections
# Collect parameters to log with Symbol keys
params_to_log = OrderedDict(
    :Date_and_Time => Dates.now(),
    :Total_Runtime => total_runtime,
    :Random_Seed_Number => random_seed,
    :Model_Architecture => string(model),
    :Optimizer => optimizer_name * "($learning_rate)",
    :Optimizer_FULL_Details => optimizer_details,
    :Batch_Size => batch_size,
    :Total_Epochs => total_epochs,
    :Best_Epoch => best_epoch,
    :Best_Training_Loss => best_training_loss,
    :Best_Validation_Loss => best_validation_loss,
    #:Huber_Loss_delta_param => delta_loss,
    :orig_Training_IDs => sort(original_ids_training),
    :orig_Test_IDs => sort(original_ids_test),
    :Sample_Ratio_Train_to_Val => "$training_count/$test_count"   
)

log_file_path = joinpath(results_dir_ANN_run, "parameters_log.txt")
log_parameters(log_file_path; params_to_log...)
######################################################################