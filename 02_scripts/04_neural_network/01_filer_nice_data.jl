### --- This part loads the parameters for training --- ###
using JSON
using DelimitedFiles

# Define the root directory of samples
root_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\01_data\\parameter_files"
#samples = "mechanical_samples\\v4"
samples = "geometrical_samples\\v6"
#samples = "all_param_samples\\v1"
root_dir = joinpath(root_dir, samples)

#### !!! MANUAL EXCEPTION OF SAMPLES !!! ###
#manual_exceptions = [57, 75, 90, 93, 99, 188, 227]#for geom v5 #[81] for geom v4 #[55] for "geom v3"
manual_exceptions = [] #for mech v4
#### !!! MANUAL EXCEPTION OF SAMPLES !!! ###

# Define the root directory for storing results
root_results_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\03_results"
results_dir = joinpath(root_results_dir, samples, "01_filtered_data")

### Define the parameters which need to be extracted:
function extract_parameters(param_data, samples)
    # Mechanical parameters
    param01 = param_data["mechanical_parameters"]["GI"]
    param02 = param_data["mechanical_parameters"]["GII,GIII"]
    param03 = param_data["mechanical_parameters"]["tI=tII=tIII"]
    param04 = param_data["mechanical_parameters"]["interface_friction"]
    param05 = param_data["mechanical_parameters"]["blade_friction"]
    # Geometrical parameters
    param06 = param_data["geometrical_parameters"]["fiber_diameter"]
    param07 = param_data["geometrical_parameters"]["droplet_diameter"]
    param08 = param_data["geometrical_parameters"]["ratio_droplet_embedded_length"]
    param09 = param_data["geometrical_parameters"]["contact_angle"]
    param10 = param_data["geometrical_parameters"]["elliptical_fiber_ratio"]
    param11 = param_data["geometrical_parameters"]["fiber_rotation"]
    param12 = param_data["geometrical_parameters"]["blade_distance"]

    # Return parameters based on the sample type
    if occursin("mechanical", samples)
        return [param01, param02, param03, param04, param05]
    elseif occursin("geometrical", samples)
        return [param06, param07, param08, param09, param10, param11, param12]
    else #
        return [param01, param02, param03, param04, param05, param06, param07, param08, param09, param10, param11, param12]
    end
end

# Create the results directory if it does not exist
if !isdir(results_dir)
    mkpath(results_dir)
    println("Created results directory at: $results_dir")
end

# Find all subdirectories of the samples (e.g., 001, 002, etc.)
subdirs = filter(d -> isdir(joinpath(root_dir, d)) && occursin(r"^\d{3}$", d), readdir(root_dir))

# Initialize containers for the selected parameters and force (or stress) - displacement data
all_params = []
all_XY_data = [] #right now the mIFSS data vs displacement is stored
exception = []

for subdir in subdirs
    # Construct the path to the parameters.json file
    param_file = joinpath(root_dir, subdir, "parameters.json")
    exception_pattern = r".*\.exception"  # Regular expression to match the exception file

    # Read and parse the JSON file
    if isfile(param_file)
        param_data = JSON.parsefile(param_file)

        # Extract parameters using the function
        extracted_params = extract_parameters(param_data, samples)

        # Store the extracted parameters as a vector of parameters
        push!(all_params, extracted_params)

    else
        println("Warning: parameters.json not found in $subdir")
        continue
    end

    # Check if any file matching the exception pattern exists in the directory
    exception_files = filter(x -> occursin(exception_pattern, x), readdir(joinpath(root_dir, subdir)))
    if !isempty(exception_files)
        push!(exception, true)
    else
        push!(exception, false)
    end
    
    # Construct paths to the load-displacement data files
    output_data_dir = joinpath(root_dir, subdir, "lhs_$(subdir)_v1_results", "outputdata")
    mifss_file = joinpath(output_data_dir, "mIFSS_from_RF2_BC3_zero_stresses_removed.txt")
    u2_file = joinpath(output_data_dir, "U2_BC3_displacement_corrected.txt")

    # Read the data files
    if isfile(mifss_file) && isfile(u2_file)
        mifss_data = readdlm(mifss_file)[:]
        u2_data = readdlm(u2_file)[:]
        
        # Store the data as a named tuple (X, Y) as Sebi did
        push!(all_XY_data, (X = u2_data, Y = mifss_data))
    else
        println("Warning: Data files not found in $output_data_dir")
    end
end

### --- Visualize existing data set --- ###
using Plots

# Define a list of line styles to iterate through
line_styles = [:solid, :dash, :dot, :dashdot]

indices_mismatch = []
# Initialize a plot object
p = plot(
    title="Original mIFSS vs. Displacement",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

# Loop through each dataset and plot it
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y

    # Check if X and Y have the same length
    if length(X) == length(Y)
        # Plot the original data
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p, X, Y, label="Sample $i", legend=:outerright)
    else
        # If they don't match, trim both arrays to the length of the shorter one
        min_length = min(length(X), length(Y))
        X = X[1:min_length]
        Y = Y[1:min_length]

        # Overwrite the data in all_XY_data
        all_XY_data[i] = (X = X, Y = Y)
        push!(indices_mismatch, i)

        println("Warning: Sample $i had mismatched lengths. Adjusted X and Y to length $min_length.")
        
        # Plot the corrected data
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p, X, Y, label="Sample $i (corrected)", legend=:outerright, linestyle=line_style)
    end
end

# Save the plot in the new results directory
plot_file = joinpath(results_dir, "00_ALL_samples.png")
savefig(p, plot_file)
display(p)

println("Original input data plot saved to: $plot_file")


### plot exception files if there are any

# Initialize a plot object
p_exception = plot(
    title="Samples with .exception",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

indices_exception = []
# Loop through each dataset and plot it
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y

    if exception[i] == true
        push!(indices_exception, i) #store indices for later filtering
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p_exception, X, Y, label="Sample $i (exception)", legend=:outerright, linestyle=line_style)
    end
end

# Save the plot in the new results directory
plot_file = joinpath(results_dir, "01_Exception_Samples.png")
savefig(p_exception, plot_file)
display(p_exception)

println("Exception samples saved to: $plot_file")


### --- Visulize filtered data --- ###

#define x_max threshold; filter out other
x_max = 0.10


# Initialize counters
global count_reach = 0
global count_no_reach = 0
# Loop through the data to count how many reach the threshold
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    if maximum(X) >= x_max
        global count_reach += 1  # Increment counter for reaching samples
    else
        global count_no_reach += 1  # Increment counter for non-reaching samples
    end
end

# Initialize two plot objects: one for samples reaching the threshold, one for those that don't
p_reach = plot(
    title="Samples Reaching x_max = 0.10: $count_reach / $(length(all_XY_data)) samples ",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

p_no_reach = plot(
    title="Samples NOT Reaching x_max = 0.10: $count_no_reach / $(length(all_XY_data)) samples",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

indices_reach = []
# Loop through each dataset and check if it reaches 95% of x_max
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y
    
    if maximum(X) >= x_max
        push!(indices_reach, i)
        # Plot the data that reaches the threshold
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p_reach, X, Y, label="Sample $i", legend=:outerright, linestyle=line_style)
    else
        # Plot the data that doesn't reach the threshold
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p_no_reach, X, Y, label="Sample $i", legend=:outerright, linestyle=line_style)
    end
end

# Save the plot in the new results directory
plot_file = joinpath(results_dir,
    "02_samles_reaching_x_max.png")
savefig(p_reach, plot_file)

plot_file = joinpath(results_dir,
    "03_samles_NOT_reaching_x_max_YET.png")
savefig(p_no_reach, plot_file)

# Display both plots
display(p_reach)
display(p_no_reach)

# Print out the counts
println("Filtered plots with x_max threshold saved")


### --- Extrapolate data to better filter out stuff --- ###

#Function to extrapolate data if the last values are close to zero (no friction)

function extrapolate_at_zero(X, Y, x_max, zero_threshold=0.1, N_last_points=3, extrap_length=10)
    # Ensure we only check the available number of points in Y
    N_to_check = min(N_last_points, length(Y))
    
    # Check if the last N Y values are close to zero and less than x_max
    if all(abs.(Y[end-N_to_check+1:end]) .<= zero_threshold) && X[end] < x_max
        # If yes, add a few new X points up to x_max
        new_xs = collect(range(X[end], x_max, length=extrap_length))
        new_ys = fill(0.0, extrap_length)  # Add corresponding Y values as zeros
        
        # Return data with added extrapolation
        return vcat(X, new_xs), vcat(Y, new_ys), true
    end
    
    # If no extrapolation needed, return empty arrays
    return [], [], false
end




all_XY_data_adp = copy(all_XY_data)




# initialize plot
p_extrapolated = plot(
    title="Extrapolated data at y=0 to x_max = $x_max",#: $count_extra / $(length(all_XY_data)) samples ",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

# x_max already defined...
indices_extrapolation_zero = []
global count_extra = 0
# Loop through each dataset and check for extrapolation
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y
    
    # Apply extrapolation
    X_extrapolated, Y_extrapolated, was_extrapolated = extrapolate_at_zero(X, Y, x_max)
    
    # Plot the extrapolated data
    if was_extrapolated
        push!(indices_extrapolation_zero, i)
        all_XY_data_adp[i] = (X = X_extrapolated, Y = Y_extrapolated)
        # Select a line style from the list (cycle through styles)
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        global count_extra += 1
        plot!(p_extrapolated, X_extrapolated, Y_extrapolated, label="Sample $i", legend=:outerright, linestyle=line_style)
    end
end

# Update the title with the count of extrapolated samples
plot_title = "Extrapolated data at y=0 to x_max = 0.10: $count_extra / $(length(all_XY_data)) samples"
title!(p_extrapolated, plot_title)

plot_file = joinpath(results_dir,
    "04_Extrapolated_samples_at_zero.png")
savefig(p_extrapolated, plot_file)

display(p_extrapolated)


### --- Extrapolate data for plateau cases with a drop greater than 50% --- ###
using Statistics
using Plots

# Function for extrapolating data with the custom criteria
function extrapolate_with_custom_criteria(X, Y, x_max; drop_threshold=0.5, relative_tolerance=0.02, extrap_length=10)
    # Maximum Y value
    Y_max = maximum(Y)

    # Identify the last X and its corresponding Y
    X_last = X[end]
    Y_last = Y[end]
    
    # Ensure the curve has dropped at least 50% from the peak value (Y_max)
    if Y_last > drop_threshold * Y_max
        return [], [], false  # No extrapolation if Y_last hasn't dropped enough
    end
    
    # Find the point where X is approximately 95% of the last X value
    X_target = 0.95 * X_last
    idx_near_target = findmin(abs.(X .- X_target))[2]  # Index of closest point to 95% of X_last
    
    # Get corresponding Y value
    Y_target = Y[idx_near_target]
    
    # Check if the Y values at these two points are within the relative tolerance
    if abs(Y_last - Y_target) / Y_max < relative_tolerance && X_last < x_max
        # Plateau detected, so extrapolate Y_last to x_max
        new_xs = collect(range(X_last, x_max, length=extrap_length))
        new_ys = fill(Y_last, extrap_length)  # Keep Y constant for plateau
        
        return vcat(X, new_xs), vcat(Y, new_ys), true
    end
    
    return [], [], false  # No extrapolation needed if criteria not met
end

### --- Plotting extrapolated data --- ###

# Initialize a plot object
p_extrapolated_custom = plot(
    title="Extrapolated Data for Plateau Detection with Custom Criteria",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm²)",
    size=(1200, 900)
)


# Counter for the number of extrapolated samples
global count_custom_extrapolated = 0
indices_extrapolation_plateau = []

# Loop through each dataset (already loaded)
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y
    
    # Apply the custom extrapolation function
    X_extrapolated, Y_extrapolated, was_extrapolated = extrapolate_with_custom_criteria(X, Y, x_max)
    
    # Plot the extrapolated data
    if was_extrapolated
        push!(indices_extrapolation_plateau, i)
        all_XY_data_adp[i] = (X = X_extrapolated, Y = Y_extrapolated)
        # Select a line style from the list (cycle through styles)
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        global count_custom_extrapolated += 1
        plot!(p_extrapolated_custom, X_extrapolated, Y_extrapolated, label="Sample $i", legend=:outerright, linestyle=line_style)
    end
end

# Update the plot title to reflect the number of extrapolated samples
plot_title_custom = "Extrapolated Data for Plateau Detection: $count_custom_extrapolated / $(length(all_XY_data)) samples"
title!(p_extrapolated_custom, plot_title_custom)

# Save the plot
plot_file_custom = joinpath(results_dir, "05_Extrapolated_samples_at_plateau.png")
savefig(p_extrapolated_custom, plot_file_custom)

# Display the plot
display(p_extrapolated_custom)

# Print out the saved file location
println("Custom extrapolated data plot saved to: $plot_file_custom")



### --- ####################### --- ###
### --- PUT TOGETHER CLEAN DATA --- ###
### --- ####################### --- ###

# Combine indices from different sources, sort, and remove duplicates
indices_clean = sort(unique(vcat(indices_reach, indices_extrapolation_zero, indices_extrapolation_plateau)))

# Define a manual exceptions list (manually specify the indices of the bad data samples)
#manual_exceptions = [55]  # Replace with the indices you want to exclude

# Remove any indices that are in the manual_exceptions list
indices_clean = filter(i -> !(i in manual_exceptions), indices_clean)

# Initialize the clean data container
clean_XY_data = []
clean_params = []

# Loop through each index in indices_clean to extract clean data
for i in indices_clean
    X = all_XY_data_adp[i].X
    Y = all_XY_data_adp[i].Y

    params = all_params[i]
    
    # Push the corresponding clean data into the clean_XY_data array
    push!(clean_XY_data, (X = X, Y = Y))
    push!(clean_params, params)
end

println("Clean data extracted for indices: $indices_clean")




### PLot clean samples
p_clean = plot(
    title="Clean samples: $(length(clean_XY_data)) / $(length(all_XY_data))",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm²)",
    size=(1200, 900)
)


for i in 1:length(clean_XY_data)
    X = clean_XY_data[i].X
    Y = clean_XY_data[i].Y

    line_style = line_styles[mod(i-1, length(line_styles)) + 1]
    plot!(p_clean, X, Y, label="Clean $i", legend=:outerright, linestyle=line_style)
end

# Save the plot
plot_file_clean = joinpath(results_dir, "06_Clean_samples.png")
savefig(p_clean, plot_file_clean)

# Display the plot
display(p_clean)

# Print out the saved file location
println("Clean samples saved to: $plot_file_clean")


### do a little bit more for ANN input...

# Function to cut the data at x_max and linearly interpolate the last y_value
function cut_and_interpolate(X, Y, x_max)
    if maximum(X) > x_max
        # Find the indices surrounding x_max
        idx_below = findlast(x -> x <= x_max, X)
        idx_above = findfirst(x -> x > x_max, X)
        
        # Linear interpolation for y at x_max
        X_below, X_above = X[idx_below], X[idx_above]
        Y_below, Y_above = Y[idx_below], Y[idx_above]
        y_interp = Y_below + (Y_above - Y_below) * (x_max - X_below) / (X_above - X_below)
        
        # Cut the data and append the interpolated point
        X = vcat(X[1:idx_below], x_max)
        Y = vcat(Y[1:idx_below], y_interp)
    end
    return X, Y
end

# Apply the cut_and_interpolate function to all clean data
for i in 1:length(clean_XY_data)
    X, Y = clean_XY_data[i].X, clean_XY_data[i].Y
    X_cut, Y_cut = cut_and_interpolate(X, Y, x_max)
    clean_XY_data[i] = (X = X_cut, Y = Y_cut)
end

println("Data cut and interpolated at x_max = $x_max")


### PLot clean samples
p_clean_cut = plot(
    title="Clean samples CUT at x_max. $(length(clean_XY_data)) / $(length(all_XY_data))",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm²)",
    size=(1200, 900)
)


for i in 1:length(clean_XY_data)
    X = clean_XY_data[i].X
    Y = clean_XY_data[i].Y

    line_style = line_styles[mod(i-1, length(line_styles)) + 1]
    plot!(p_clean_cut, X, Y, label="Clean $i", legend=:outerright, linestyle=line_style)
end

# Save the plot
plot_file_clean_cut = joinpath(results_dir, "07_Clean_samples_CUT_x_max.png")
savefig(p_clean_cut, plot_file_clean_cut)

# Display the plot
display(p_clean_cut)

# Print out the saved file location
println("Clean samples saved to: $plot_file_clean_cut")


### Resample data even more for input for Sebis ANN SCRIPT
using Interpolations
using SavitzkyGolay #to smooth out the linestyle

window_size = 5  # Must be odd
polynomial_order = 1  # Must be less than window_size

# Resample using linear interpolation
Xs = range(0, x_max, length=500)
Ys = map(clean_XY_data) do d
    f = linear_interpolation(d.X, d.Y)
    Y_interp = f.(Xs)
    # Sightly smooth using Savitzky-Golay filter
    sg = savitzky_golay(Y_interp, window_size, polynomial_order)
    Y_smooth = sg.y  # The filtered signal is stored in sg.y
    return Y_smooth
end


# Initialize plot objects for comparison
p_clean_cut_resample = plot(
    title="FINAL input for ANN: $(length(clean_XY_data)) / $(length(all_XY_data))",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm²)",
    size=(1200, 900)
)

# Plot the resampled data
for i in 1:length(Ys)
    line_style = line_styles[mod(i-1, length(line_styles)) + 1]
    plot!(p_clean_cut_resample, Xs, Ys[i], label="Clean $i", legend=:outerright, linestyle=line_style)
end

# Display the plot
display(p_clean_cut_resample)

plot_file_clean_cut_re = joinpath(results_dir, "08_Final_input_ANN.png")
savefig(p_clean_cut_resample, plot_file_clean_cut_re)


#############################
### SAVE CLEAN PARAMETERS ###
#############################
using JLD2
# Function to save Xs, Ys, and clean_params to a .jld2 file
function save_data_to_jld2(filename, Xs, Ys, clean_params, indices_clean)
    @save filename Xs Ys clean_params indices_clean
    println("Data saved to $filename")
end

# Example usage
save_data_file = joinpath(results_dir, "clean_data.jld2")
save_data_to_jld2(save_data_file, Xs, Ys, clean_params, indices_clean)
