using JLD2
using BSON
using Dates 
using Interpolations
using Flux
using Random
using Statistics
using Distributions
#using Plots
using ProgressMeter
using Printf
## new
using FileIO
using CairoMakie
using LinearAlgebra
using Trapz

#sample_versions = ["geometrical_samples\\v6"]
sample_versions = ["mechanical_samples\\v4finer"]
#sample_versions = ["selected_param_samples2\\v1", "selected_param_samples2\\v2"]

total_epochs = 10000
delta_loss = 0.001 #0.001 #this value works great for geometrical_samples\\v6 (shifts basically to mean absolute error)
random_seed = 1234
Random.seed!(random_seed)
description = "Maximum force"

# Define parameter names based on the sample type (for PLOTS)
mech_params_names = ["GI", "GII", "t", "i_fric", "b_fric"]
geom_params_names = ["fd", "D", "L/D", "θ", "ell/r", "φ", "bl_d"] # θ is contact angle; φ is fiber rotation
#selected_params01_names = ["GII", "t", "fd", "D", "bl_d"]
selected_params02_names = ["GII", "t", "fd", "ell/r", "bl_d"]

### ------ Filestructure and Dir creation ------ ###
root_results_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\03_results" #results folder

common_prefix = dirname(sample_versions[1])
version_names = map(basename, sample_versions)
combined_name = joinpath(common_prefix,join(version_names, "_"))
results_dir = joinpath(root_results_dir, combined_name, "01_filtered_data")
results_dir_ANN = joinpath(root_results_dir, combined_name, "03_ANN_hessian")

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



# Use maximum stress value for training
Ys_combined_max_stress = map(Ys_combined) do Y
    maximum(Y)
end

# Compose training and test sets (input, label) with max stress values
data_training = [ (normalized_params[id], Ys_combined_max_stress[id]) for id in training_ids]
data_test = [ (normalized_params[id], Ys_combined_max_stress[id]) for id in test_ids]

# Determine size of layers
N_inp = length(first(clean_params_combined))
N_out_max = 1  # Since max stress is a scalar

model = Chain(
    Dense(N_inp => 4*N_inp, celu),
    #Dense(4*N_inp => 4*N_inp, celu),
    Dense(4*N_inp => N_out_max)
) |> f64

# This is the batch loader with the minibatch size
batch_size = 4
batches = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true) 

learning_rate = 0.0001
optim = Flux.setup(Flux.Adam(learning_rate), model)


function train_max_stress_model(total_epochs)
    # Initialize variables
    start_time = Dates.now()
    losses_training = Float64[]
    losses_validation = Float64[]
    best_validation_loss = Inf
    best_epoch = 0
    best_model = Flux.state(model)
    
    # Initialize progress bar
    p = Progress(total_epochs, desc="Training Max Stress Model")

    # Start training loop
    for epoch in 1:total_epochs
        # Loop over minibatches
        for batch in batches
            val, grads = Flux.withgradient(model) do m
                mean(Flux.Losses.huber_loss(m(input), label; delta=delta_loss) for (input, label) in batch)
            end
            # Update the model using the gradients
            Flux.update!(optim, model, grads[1])
        end

        # Compute current training and validation losses
        current_training_loss = mean(Flux.Losses.huber_loss(model(x), y; delta=delta_loss) for (x, y) in data_training)
        current_validation_loss = mean(Flux.Losses.huber_loss(model(x), y; delta=delta_loss) for (x, y) in data_test)

        # Update progress bar with current losses
        ProgressMeter.update!(p, epoch; showvalues = [(:Train_Loss, round(current_training_loss, digits=8)),
                                                      (:Val_Loss, round(current_validation_loss, digits=8))])

        # Store the losses for future reference
        push!(losses_training, current_training_loss)
        push!(losses_validation, current_validation_loss)

        # Save the best model if validation loss improves
        if current_validation_loss < best_validation_loss
            best_validation_loss = current_validation_loss
            best_epoch = epoch
            best_model = Flux.state(model)
        end
    end

    finish!(p)  # Finalize the progress bar

    # Load the best model
    Flux.loadmodel!(model, best_model)

    # Print summary of training results
    println("\nTraining complete.")
    println("Best validation loss at epoch $best_epoch with loss $best_validation_loss")

    total_runtime = Dates.now() - start_time

    return best_model, losses_training, losses_validation, best_epoch, best_validation_loss, total_runtime
end

best_model, losses_training, losses_validation, best_epoch, best_validation_loss, total_runtime = train_max_stress_model(total_epochs)
best_training_loss = losses_training[best_epoch]

# Save the model after training
model_file = joinpath(results_dir_ANN_run, "trained_model.bson")
BSON.@save model_file model
println("Trained model saved to $model_file")

# Plot losses
save_plot_loss_log_max_stress = joinpath(results_dir_ANN_run, "loss_f_log_max_stress.png")
fig = Figure(size = (1200, 900))
ax = Axis(fig[1, 1], xlabel = "epochs", ylabel = "MSE", yscale = log10)
lines!(ax, 1:length(losses_training), losses_training, label = "training loss for max stress", color=:blue)
lines!(ax, 1:length(losses_validation), losses_validation, label = "validation loss for max stress", color=:green)
vlines!(ax, [best_epoch], label = "Best Epoch", linestyle=:dash, color=:red)
axislegend(ax)
save(save_plot_loss_log_max_stress, fig)
display(fig)


# Compute the losses for each test sample
test_losses = [Flux.Losses.huber_loss(model(d[1]), d[2]; delta=delta_loss) for d in data_test]
n_min = sort(collect(1:length(test_losses)), by=i->test_losses[i])
n_max = sort(collect(1:length(test_losses)), by=i->test_losses[i], rev=true) # best losses
middle_start = Int(floor(length(test_losses) / 2)) - 3  # Start 3 positions before the middle
middle_end = middle_start + 5  # Take 6 samples
n_middle = n_max[middle_start:middle_end]

function plot_comparisons(sample_indices, title_str, save_name)
    # Extract the original IDs, truth, and predictions for the selected samples
    sample_ids = [original_ids_test[idx] for idx in sample_indices]
    predictions = [model(data_test[idx][1])[1] for idx in sample_indices]
    truths = [data_test[idx][2] for idx in sample_indices]

    # Prepare the data for the bar plot
    samples = [1,1,2,2,3,3,4,4,5,5,6,6]  # Corresponding to 3 samples (truth and prediction for each)
    bar_values = [truths[1], predictions[1], truths[2], predictions[2], truths[3], predictions[3], truths[4], predictions[4], truths[5], predictions[5], truths[6], predictions[6]]

    # Plot
    fig = barplot(samples, bar_values,
        dodge = [1,2,1,2,1,2,1,2,1,2,1,2],  # Dodging for truth (1) and predictions (2)
        color = [:blue, :green, :blue, :green, :blue, :green, :blue, :green, :blue, :green, :blue, :green],  # Colors for truth and prediction
        axis = (xticks = (1:6, ["orig ID: $(sample_ids[1])", "orig ID: $(sample_ids[2])", "orig ID: $(sample_ids[3])", "orig ID: $(sample_ids[4])", "orig ID: $(sample_ids[5])", "orig ID: $(sample_ids[6])"]),
                title = title_str)
        )
    
    # Set y-axis limits
    ylims!(fig.axis, (0, 20))

    # Save the figure
    save(save_name, fig)
    
    display(fig)
end

# Example usage for best, worst, and median samples
plot_comparisons(n_min[1:6], "Best Samples", joinpath(results_dir_ANN_run, "n_best.png"))
plot_comparisons(n_max[1:6], "Worst Samples", joinpath(results_dir_ANN_run, "n_worst.png"))
plot_comparisons(n_middle, "Median Samples", joinpath(results_dir_ANN_run, "n_median.png"))





############# Get parameter names for Plotting #############
# Decide which parameter names to use based on the common prefix (geometrical_sampls, mechanical_samples, etc.)
if occursin("mechanical_samples", common_prefix)
    param_names = mech_params_names
elseif occursin("geometrical_samples", common_prefix)
    param_names = geom_params_names
elseif occursin("selected_param_samples", common_prefix)
    param_names = selected_params02_names
else
    param_names = vcat(mech_params_names, geom_params_names)
end

parameter_ranges = extrema.(eachrow(reduce(hcat, clean_params_combined)))
range_text = "Parameter ranges:\n " * join([ "$(param_names[i]): (" * string(round(l, digits=4)) * "-" * string(round(u, digits=4)) * ")" 
                                           for (i, (l, u)) in enumerate(parameter_ranges)], ", ")

function generate_param_string(sample_idx)
    params = clean_params_combined_test[sample_idx]
    param_str = join([ string(param_names[i]) * ": " * string(round(p, digits=4)) for (i, p) in enumerate(params)], ", ")
    return param_str
end     
####### 




##### SEBIS STUFF #####

# Plotting
f_NN = Figure(size=(1000,1000))

ax_Taylor = Axis(f_NN[2,2:4], xlabel="taylor", ylabel="model", title="Taylor series approximation")
ax_gradient = Axis(f_NN[3:4,1], xlabel="Gradient", yticks = (1:N_inp, param_names), title="Gradient")
ax_hessian = Axis(f_NN[3:4,2:3], xticks = (1:N_inp, param_names),xticklabelrotation=π/2, title="Hessian")
linkyaxes!(ax_gradient, ax_hessian)
hideydecorations!(ax_hessian, ticks = false)

g_conv = GridLayout(f_NN[1,1:4])
ax_NN = Axis(g_conv[1,1:3], xlabel="epochs", ylabel="MSE", yscale=log10, title="Convergence")
ax_Worst = Axis(f_NN[2,1], xlabel="truth", ylabel="model", title="Testset")

Label(f_NN[end,1, BottomLeft()], description, fontsize = 20, halign = :left, valign=:bottom)

lines!(ax_NN, losses_training, label="training")
lines!(ax_NN, losses_validation, label="validation")

Legend(g_conv[1,4], ax_NN)

scatter!(ax_Worst, [d[2][1] for d in data_test], [model(d[1])[1] for d in data_test])
#l_ext = collect(extrema([d[2][1] for d in data_test]))
lines!(ax_Worst, [5,20], [5,20], color=:black)
xlims!(ax_Worst, 5, 20)
ylims!(ax_Worst, 5, 20)


function taylor_polynomial(model, p0)
    a0 = model(p0)[1]
    g = Flux.gradient(x->model(x)[1], p0)[1]
    H = Flux.hessian(x->model(x)[1], p0)
    return function (x)
        Δx = x-p0
        return a0+(Δx⋅g)
    end, function (x)
        Δx = x-p0
        return a0+Δx⋅g+((H*Δx)⋅Δx)*1/2
    end, g, H
end

# Location to develop taylor series around
p0 = [1.0 for _ in 1:N_inp]
tpo1, tpo2, g, H = taylor_polynomial(model, p0)

base = [model(ps |> collect)[1] for ps in Iterators.product([range(0,1,length=5) for _ in 1:N_inp]...)][:];
predo1 = [tpo1(ps|>collect) for ps in Iterators.product([range(0,1,length=5) for _ in 1:N_inp]...)][:];
predo2 = [tpo2(ps|>collect) for ps in Iterators.product([range(0,1,length=5) for _ in 1:N_inp]...)][:];

scatter!(ax_Taylor, base, predo1, label="o1", marker=:x, markersize=5.0)
scatter!(ax_Taylor, base, predo2, label="o2", marker=:x, markersize=5.0)
lines!(ax_Taylor, [5,20], [5,20], color=:black)
axislegend(ax_Taylor, position=:rb)
xlims!(ax_Taylor, 5, 20)
ylims!(ax_Taylor, 5, 20)

barplot!(ax_gradient, g, direction=:x)
xlims!(ax_gradient, 0, 15)

H_ext = maximum(abs.(H))
H_ext = H_ext ≈ 0.0 ? 1.0 : H_ext
hm = heatmap!(ax_hessian,H, colormap=:RdBu, colorrange=(-H_ext,H_ext))
Colorbar(f_NN[3:4, 4], hm)

display(f_NN)

save_sebi_plot = joinpath(results_dir_ANN_run, "Grad100.png")
save(save_sebi_plot, f_NN)

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
    :Huber_Loss_delta_param => delta_loss,
    :orig_Training_IDs => sort(original_ids_training),
    :orig_Test_IDs => sort(original_ids_test),
    :Sample_Ratio_Train_to_Val => "$training_count/$test_count"   
)

log_file_path = joinpath(results_dir_ANN_run, "parameters_log.txt")
log_parameters(log_file_path; params_to_log...)
######################################################################