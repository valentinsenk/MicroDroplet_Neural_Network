using Pkg; Pkg.activate(".")
# Generate training data
# Some 5 parameter function f(x)
parameter_ranges = [
    (-4,4),
    (-4,4),
    (-4,4),
    (-10, 10),
]
# sample_function(x, params) = sum(p*x^(i-1) for (i,p) in enumerate(params))
sample_function(x, params) = (x-10+params[1])*(x-40+params[2])*(x-70+params[3])+20*params[4]*x
N_functions = 100
params = [map(((l,u),)->l+rand()*(u-l),parameter_ranges) for _ in 1:N_functions]
data = map(params) do p
    (X=collect(0:0.1:100), Y=sample_function.(0:0.1:100, Ref(p)))
end
# Data convention:
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

# Prepare input data
# Resample input functions
N_samples = 100
# Get largest smallest and smallest largest X
x_min = maximum(minimum.((d->d.X).(data)))
x_max = minimum(maximum.((d->d.X).(data)))

# Resample using linear interpolation
Xs = range(x_min, x_max, length=100)
Ys = map(data) do d
    f = linear_interpolation(d.X, d.Y)
    f.(Xs)
end

# Normalize input parameters
parameter_ranges_from_data = extrema.(eachrow(reduce(hcat, params)))
normalized_params = map(params) do p
    [(p[i]-l)/(u-l) for (i,(l,u)) in enumerate(parameter_ranges_from_data)]
end

# Train the neural network
# Select training and test data
N_training = Int(ceil(length(Ys)/3*2))
N_test = length(Ys)-N_training
# Randomize IDs for trying different sets
training_ids = Set(shuffle(1:length(Ys))[1:N_training])
test_ids = setdiff(Set(1:length(Ys)),training_ids)

# Compose training and test sets (input, label)
data_training = [ (normalized_params[id], Ys[id]) for id in training_ids]
data_test = [ (normalized_params[id], Ys[id]) for id in test_ids]

# Determine size of layers
N_inp = length(first(params))
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
    Dense(4*N_inp => 4*N_inp, relu),
    Dense(4*N_inp => N_out)
) |> f64

# This is the optimizer
optim = Flux.setup(Flux.Adam(0.1), model)

# This is the batch loader with the minibatch size
batches = Flux.DataLoader(data_training, batchsize=4, shuffle=true)

# Init variables for tracking the loss through the epochs
losses_training = Float64[]
losses_test = Float64[]

# Train for 5000 epochs
for epoch in 1:5000
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

# Plotting the loss
plot([losses_training, losses_test], label=["training" "test"], yscale=:log10, xlabel="epochs", ylabel="MSE")
savefig("fig/Example_01/loss_f.png")

# Plotting the N worst approximations
test_losses = [Flux.Losses.mse(model(d[1]),d[2]) for d in data_test]
n_max = sort(collect(1:length(test_losses)), by=i->test_losses[i], rev=true)
plot(layout=grid(2,3),[plot(Xs,[model(data_test[idx][1]),data_test[idx][2]], labels=["prediction" "truth"]) for idx in n_max[1:6]]...)
savefig("fig/Example_01/n_worst.png")

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
for (idx,v) in enumerate(["p$i" for i in 1:N_inp])
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
qtly = quantile.(eachrow(_base),Ref([0.25,0.75]))
_base_min = (x->x[1]).(qtly)
_base_max = (x->x[2]).(qtly)
_base_mean = mean.(eachrow(_base))

# Plot inner quartile range of base values
plt1 = plot(Xs, _base_min, lw=0, fillrange=_base_max, alpha=0.5, color=:blue, label=:none, ylabel="Function")
plot!(plt1, Xs, _base_mean, color=:blue, label=:none)

# Combine both plots
plot(layout=grid(2,1), plt1, plt2)
savefig("fig/Example_01/sensitivity.png")