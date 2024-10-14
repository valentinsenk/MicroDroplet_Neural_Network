### Load data from filtered script ###
using JLD2
using Dates 

# List of samples to use (e.g., v4 and v5)
#sample_versions = ["geometrical_samples\\v4", "geometrical_samples\\v5"]
#sample_versions = ["mechanical_samples\\v4"]
sample_versions = ["geometrical_samples\\v6"]
# Create a combined name for the result folder
common_prefix = dirname(sample_versions[1])
version_names = map(basename, sample_versions)
combined_name = joinpath(common_prefix,join(version_names, "_"))

# Define parameter names based on the sample type (for PLOTS)
mech_params_names = ["GI", "GII", "t", "i_fric", "b_fric"]
geom_params_names = ["fd", "D", "D/L", "θ", "ell/r", "φ", "bl_d"] # θ is contact angle; φ is fiber rotation

root_results_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\03_results"
results_dir = joinpath(root_results_dir, combined_name, "01_filtered_data")
results_dir_ANN = joinpath(root_results_dir, combined_name, "02_ANN")

# Function to load Xs, Ys, and clean_params from a .jld2 file
function load_data_from_jld2(filename)
    @load filename Xs Ys clean_params indices_clean
    println("Data loaded from $filename")
    return Xs, Ys, clean_params, indices_clean
end

# Load data from more versions
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


# Create the results directory FOR ANN if it does not exist
if !isdir(results_dir_ANN)
    mkpath(results_dir_ANN)
    println("Created results directory at: $results_dir_ANN")
end

using Printf 
# Function to create a new run directory
function create_new_run_dir(base_dir::String)
    # Find existing run directories
    existing_runs = filter(name -> occursin(r"^run\d{2}$", name), readdir(base_dir))
    # Extract run numbers
    run_numbers = parse.(Int, [match(r"^run(\d{2})$", name).captures[1] for name in existing_runs])
    # Determine the next run number
    next_run_number = isempty(run_numbers) ? 1 : maximum(run_numbers) + 1
    # Format the run directory name
    run_dir_name = @sprintf("run%02d", next_run_number)
    # Create the full path
    run_dir = joinpath(base_dir, run_dir_name)
    # Create the directory
    mkpath(run_dir)
    println("Created new run directory at: $run_dir")
    return run_dir
end

# Use the function to create the run directory
results_dir_ANN_run = create_new_run_dir(results_dir_ANN)


# Data convention (following Sebis template):
# - The XY data for the load displacement plot is a list of named tuples with entries .X and .Y.
# - The respective parameters are in a vector of vectors with the subvector as [p1, p2, p3, ...]
# While the test data is evenly sampled, for the sake of generality of this code, it is assumed
# to be sampled unevenly.

using Interpolations
using Flux
using Random
using Statistics
using Distributions
using Plots

### Plot new data for test reasons
# Create a plot with index numbering as the legend
#plot(title="Resampled Data", xlabel="X", ylabel="Y")


#for i in 1:length(Ys)
#    plot!(Xs, Ys[i], label="Sample $i")
#end

# Add a legend outside the plot
#plot!(legend=:outerright)


# Normalize input parameters
parameter_ranges_from_data = extrema.(eachrow(reduce(hcat, clean_params_combined)))
normalized_params = map(clean_params_combined) do p
    [(p[i]-l)/(u-l) for (i,(l,u)) in enumerate(parameter_ranges_from_data)]
end

# Train the neural network
# Select training and test data
N_training = Int(ceil(length(Ys_combined)/3*2))
#N_test = length(Ys_combined)-N_training

## original
# Randomize IDs for trying different sets
#
training_ids = Set(shuffle(1:length(Ys_combined))[1:N_training])
test_ids = setdiff(Set(1:length(Ys_combined)),training_ids)
# Compose training and test sets (input, label)
data_training = [ (normalized_params[id], Ys_combined[id]) for id in training_ids]
original_ids_training = [ indices_combined[id] for id in training_ids]  # Store original IDs for training data
clean_params_combined_training = [ clean_params_combined[id] for id in training_ids]

data_test = [ (normalized_params[id], Ys_combined[id]) for id in test_ids]
original_ids_test = [ indices_combined[id] for id in test_ids]  # Store original IDs for test data
clean_params_combined_test = [ clean_params_combined[id] for id in test_ids]

# Determine size of layers
N_inp = length(first(clean_params_combined))
N_out = length(Xs)

# Define the model.
# This model worked quite well for this specific problem.
# However, the hyperparameters
# - Number of layers
# - Size of layers
# - Activation functions (https://fluxml.ai/Flux.jl/stable/reference/models/activation/)
# - Optimizers and learning rates (https://fluxml.ai/Flux.jl/stable/reference/training/optimisers/)
# - Minibatch size 
# - Number of epochs
# need to be assessed for each new problem.

# This is the actual model
# The reul function is just added to highlight usage of activation functions.
# The original model is linear in the parameter space so an ANN without an activation function
# would be sufficient.
model = Chain(
    Dense(N_inp => 4*N_inp, relu),
    #Dropout(0.2), # to reduce overfit
    #LayerNorm(4*N_inp, relu),
    #Dense(4*N_inp => 4*N_inp, relu),
    #Dropout(0.2),
    #LayerNorm(4*N_inp, relu),
    Dense(4*N_inp => N_out)
) |> f64

# This is the optimizer
learning_rate = 0.001
optim = Flux.setup(Flux.Adam(learning_rate), model)

# This is the batch loader with the minibatch size
batch_size = 8
batches = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true) #batchsize=2 is worse for mechsamples...

# Init variables for tracking the loss through the epochs
losses_training = Float64[]
losses_test = Float64[]



# Train for 5000 epochs
total_epochs = 10000
for epoch in 1:total_epochs
    # Print the current epoch dynamically on the same line
    print("\rTraining epoch $epoch / $total_epochs ...")
    flush(stdout)
    # Loop over minibatches
    for batch in batches
        # Compute average loss and gradient for the minibatch
        val, grads = Flux.withgradient(model) do m
            mean(Flux.Losses.mse(m(input),label) for (input,label) in batch)
        end
        # Update the model with the average minibatch gradient
        Flux.update!(optim, model, grads[1])
    end
    # Store the information about the current loss function
    push!(losses_training,mean(Flux.Losses.mse(model(x),y) for (x,y) in data_training))
    push!(losses_test,mean(Flux.Losses.mse(model(x),y) for (x,y) in data_test))
end

# Make sure the final epoch number prints after the loop
println("\nTraining complete.")


# Decide which parameter names to use based on the common prefix (geometrical_sampls, mechanical_samples, etc.)
if occursin("mechanical_samples", common_prefix)
    param_names = mech_params_names
elseif occursin("geometrical_samples", common_prefix)
    param_names = geom_params_names
else
    param_names = vcat(mech_params_names, geom_params_names)
end

#parameter range information for plots
parameter_ranges = extrema.(eachrow(reduce(hcat, clean_params_combined)))
range_text = "Parameter ranges:\n " * join([ "$(param_names[i]): (" * string(round(l, digits=4)) * "-" * string(round(u, digits=4)) * ")" 
                                           for (i, (l, u)) in enumerate(parameter_ranges)], ", ")


##### LOG FILE for parameters #####
# Collect parameters to log with Symbol keys
params_to_log = Dict(
    :Date_and_Time => Dates.now(),
    :Model_Architecture => string(model),
    :Optimizer => "Adam($learning_rate)",
    :Batch_Size => batch_size,
    :Total_Epochs => total_epochs,
    :orig_Training_IDs => sort(original_ids_training),
    :orig_Test_IDs => sort(original_ids_test),
    :Parameter_Ranges => parameter_ranges
)

# Function to log parameters to a file
function log_parameters(filepath::String; kwargs...)
    open(filepath, "w") do io
        println(io, "Model and Training Parameters:")
        for (key, value) in kwargs
            println(io, "$key: $value")
        end
    end
    println("Parameters logged to $filepath")
end

# Log file path
log_file_path = joinpath(results_dir_ANN_run, "parameters_log.txt")

# Log the parameters
log_parameters(log_file_path; params_to_log...)

###############################################################

save_plot_loss_log = joinpath(results_dir_ANN_run, "loss_f_log.png")
save_plot_loss = joinpath(results_dir_ANN_run, "loss_f.png")
save_plot_worst = joinpath(results_dir_ANN_run, "n_worst.png")
save_plot_best = joinpath(results_dir_ANN_run, "n_best.png")
save_plot_median = joinpath(results_dir_ANN_run, "n_median.png")



# Plotting the loss
p_loss_log = plot([losses_training, losses_test], label=["training" "test"], yscale=:log10, xlabel="epochs", ylabel="MSE", size=(1200, 900))
savefig(save_plot_loss_log)
display(p_loss_log)

p_loss = plot([losses_training, losses_test], label=["training" "test"], xlabel="epochs", ylabel="MSE", size=(1200, 900))
#save_plot_loss = joinpath(results_dir_ANN, "loss_f.png")
savefig(save_plot_loss)
display(p_loss)

### --- make some nice plots --- ###


function generate_param_string(sample_idx)
    params = clean_params_combined_test[sample_idx]
    param_str = join([ string(param_names[i]) * ": " * string(round(p, digits=4)) for (i, p) in enumerate(params)], ", ")
    return param_str
end


# Plotting the N worst approximations
test_losses = [Flux.Losses.mse(model(d[1]),d[2]) for d in data_test]
n_max = sort(collect(1:length(test_losses)), by=i->test_losses[i], rev=true)
p_worst = plot(layout=grid(2,3), plot_title=range_text, plot_titlefontsize=6,
    [plot(Xs, [model(data_test[idx][1]), data_test[idx][2]], 
    labels=["prediction" "truth"], 
    title="WORST SAMPLE [original ID: $(original_ids_test[idx])]\n$(generate_param_string(idx))", size=(1200, 900),
        titlefont=5, ylims=(0, 20)) for idx in n_max[1:6]]...)
#save_plot_worst = joinpath(results_dir_ANN, "n_worst.png")
savefig(p_worst, save_plot_worst)
display(p_worst)


# Plotting the N best approximations
n_min = sort(collect(1:length(test_losses)), by=i->test_losses[i])  # Sort in ascending order for the best losses
p_best = plot(layout=grid(2, 3), plot_title=range_text, plot_titlefontsize=6,
    [plot(Xs, [model(data_test[idx][1]), data_test[idx][2]], 
    labels=["prediction" "truth"], 
    title="BEST SAMPLE [original ID: $(original_ids_test[idx])]\n$(generate_param_string(idx))", size=(1200, 900),
        titlefont=5, ylims=(0, 20)) for idx in n_min[1:6]]...)
#save_plot_best = joinpath(results_dir_ANN, "n_best.png")
savefig(save_plot_best)
display(p_best)

# Plotting samples close to the median of all samples 
middle_start = Int(floor(length(test_losses) / 2)) - 3  # Start 3 positions before the middle
middle_end = middle_start + 5  # Take 6 samples
n_middle = n_max[middle_start:middle_end]
p_median = plot(layout=grid(2, 3), plot_title=range_text, plot_titlefontsize=6,
    [plot(Xs, [model(data_test[idx][1]), data_test[idx][2]], 
    labels=["prediction" "truth"], 
    title="MEDIAN SAMPLE [original ID: $(original_ids_test[idx])]\n$(generate_param_string(idx))", size=(1200, 900),
        titlefont=5, ylims=(0, 20)) for idx in n_middle]...)
#save_plot_median = joinpath(results_dir_ANN, "n_median.png")
savefig(save_plot_median)
display(p_median)


using BSON
# Save the model after training
model_file = joinpath(results_dir_ANN_run, "trained_model.bson")
BSON.@save model_file model
println("Trained model saved to $model_file")





###
### UNTIL HERE FIRST TESTS DONE
###

plt = plot()

# Compute the range of function values
base = [model(collect(ps)) for ps in Iterators.product([range(0,1, length=3) for _ in 1:N_inp]...)]
# Compute the derivative for all parameters at all locations in base
sens = [Flux.jacobian(model, collect(ps))[1] for ps in Iterators.product([range(0,1, length=3) for _ in 1:N_inp]...)]

# Fit normal distributions to the gradient data
dists = [[fit(Normal,[sens[s][i,pidx] for s in eachindex(sens)]) for i in 1:N_out] for pidx in 1:N_inp]

# Plot the gradient data
colors = palette(:rainbow, N_inp)
plt2 = plot(ylabel="Gradient")
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
display(plt2)

# Compute visualization of base values
_base = reduce(hcat,base)
qtly = quantile.(eachrow(_base),Ref([0.25,0.75]))
_base_min = (x->x[1]).(qtly)
_base_max = (x->x[2]).(qtly)
_base_mean = mean.(eachrow(_base))

# Plot inner quartile range of base values
plt1 = plot(Xs, _base_min, lw=0, fillrange=_base_max, alpha=0.5, color=:blue, label=:none, ylabel="Function")
plot!(plt1, Xs, _base_mean, color=:blue, label=:none)

# Combine both plots
p_gradient = plot(layout=grid(2,1), plt1, plt2, size=(1200, 900))
save_gradient = joinpath(results_dir_ANN_run, "gradient.png")

savefig(save_gradient)
display(p_gradient)