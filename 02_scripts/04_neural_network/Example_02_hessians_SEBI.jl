using Pkg; Pkg.activate(".")
using Interpolations
using Flux
using FileIO
using CairoMakie
using Trapz
using Random
using Statistics
using LinearAlgebra

raw_data = load(joinpath("data","data.jld2"))

data = raw_data["LDs"]
params = raw_data["parameter_values"]
parameter_names = raw_data["parameter_names"]

# Set seed for training
Random.seed!(1234)

# Prepare input data
# Option 1 - maximum value
# Ys = map(data) do d
#     maximum(d.y)
# end
# description = "Maximum force"

# Option 2 - potential of external forces
# Ys = map(data) do d
#     trapz(d.x, d.y)
# end
# description = "ext. Potential"

# Option 3 - initial stiffness
# strongly depends on the problem
# how difficult it is to obtain this value in a robuts manner
Ys = map(data) do d
    d.y[2]/d.x[2]
end
description = "LE stiffness"

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
N_out = length(first(Ys))

# Define the model.
model = Chain(
    Dense(N_inp => 4*N_inp, celu),
    Dense(4*N_inp => 4*N_inp, celu),
    Dense(4*N_inp => N_out)
) |> f64

# This is the optimizer
optim = Flux.setup(Flux.Adam(), model)

# This is the batch loader with the minibatch size
batches = Flux.DataLoader(data_training, batchsize=4, shuffle=true)

# Init variables for tracking the loss through the epochs
losses_training = Float64[]
losses_test = Float64[]

# Store best (in terms of validation set) model here
best_model = Flux.state(model)
# Train for 5000 epochs
for epoch in 1:10000
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
    # Store best
    if length(losses_test) == 1 || (losses_test[end] < minimum(losses_test[1:end-1]))
        best_model = Flux.state(model)
    end
end

# Loading best model
Flux.loadmodel!(model, best_model)

# Plotting
f_NN = Figure(size=(1000,1000))

ax_Taylor = Axis(f_NN[2,2:4], xlabel="taylor", ylabel="model", title="Taylor series approximation")
ax_gradient = Axis(f_NN[3:4,1], xlabel="Gradient", yticks = (1:N_inp, parameter_names), title="Gradient")
ax_hessian = Axis(f_NN[3:4,2:3], xticks = (1:N_inp, parameter_names),xticklabelrotation=π/2, title="Hessian")
linkyaxes!(ax_gradient, ax_hessian)
hideydecorations!(ax_hessian, ticks = false)

g_conv = GridLayout(f_NN[1,1:4])
ax_NN = Axis(g_conv[1,1:3], xlabel="epochs", ylabel="MSE", yscale=log10, title="Convergence")
ax_Worst = Axis(f_NN[2,1], xlabel="truth", ylabel="model", title="Testset")

Label(f_NN[end,1, BottomLeft()], description, fontsize = 20, halign = :left, valign=:bottom)

lines!(ax_NN, losses_test, label="test")
lines!(ax_NN, losses_training, label="training")

Legend(g_conv[1,4], ax_NN)

scatter!(ax_Worst, [d[2][1] for d in data_test], [model(d[1])[1] for d in data_test])
l_ext = collect(extrema([d[2][1] for d in data_test]))
lines!(ax_Worst, l_ext, l_ext, color=:black)

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
p0 = [0.5 for _ in 1:N_inp]
tpo1, tpo2, g, H = taylor_polynomial(model, p0)

base = [model(ps |> collect)[1] for ps in Iterators.product([range(0,1,length=5) for _ in 1:N_inp]...)][:];
predo1 = [tpo1(ps|>collect) for ps in Iterators.product([range(0,1,length=5) for _ in 1:N_inp]...)][:];
predo2 = [tpo2(ps|>collect) for ps in Iterators.product([range(0,1,length=5) for _ in 1:N_inp]...)][:];

scatter!(ax_Taylor, base, predo1, label="o1", marker=:x, markersize=5.0)
scatter!(ax_Taylor, base, predo2, label="o2", marker=:x, markersize=5.0)
lines!(ax_Taylor, collect(extrema(base)), collect(extrema(base)), color=:black)
axislegend(ax_Taylor, position=:rb)

barplot!(ax_gradient, g, direction=:x) 

H_ext = maximum(abs.(H))
H_ext = H_ext ≈ 0.0 ? 1.0 : H_ext
hm = heatmap!(ax_hessian,H, colormap=:RdBu, colorrange=(-H_ext,H_ext))
Colorbar(f_NN[3:4, 4], hm)

save(joinpath("fig", "Example_02", replace(description, " "=>"_")*".png"),f_NN)