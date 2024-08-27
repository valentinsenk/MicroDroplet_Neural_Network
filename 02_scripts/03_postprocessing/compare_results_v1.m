close all

% Define the root directory where the parameter directories are located
root_dir = '/home/vsenk/Droplet_Tests_FEA/01_neural_network_project/01_data/parameter_files/mechanical_samples/v1/';

% Get the result folders
result_folders = dir(fullfile(root_dir, '*_results'));

% Create a directory for comparison results if it doesn't exist
compare_folder = fullfile(root_dir, 'compare_allresults');
if ~exist(compare_folder, 'dir')
    mkdir(compare_folder);
end

% Define a list of line styles
line_styles = {'-', '--', ':', '-.'};
line_colors = lines(length(result_folders)); % Get a colormap with a unique color for each variable

% Plot and save original data
figure;
hold on;
title('Force-Displacement Comparison');
xlabel('Displacement (mm)');
ylabel('Force (N)');
legend_entries = {};

for i = 1:length(result_folders)
    try
        folder_name = result_folders(i).name;
        folder_path = fullfile(result_folders(i).folder, folder_name);
        data_folder = fullfile(folder_path, 'outputdata');

        % Specific RF2 and U2 filenames
        rf2_file_name = 'RF2_BC3.txt';
        u2_file_name = 'U2_BC3.txt';

        % Read data from files
        rf2_data = read_data(fullfile(data_folder, rf2_file_name));
        u2_data = read_data(fullfile(data_folder, u2_file_name));

        % Determine the line style and color
        line_style = line_styles{mod(i-1, length(line_styles)) + 1};
        line_color = line_colors(i, :);

        % Plot the original data
        plot(u2_data, rf2_data, 'LineWidth', 1, 'LineStyle', line_style, 'Color', line_color);

        % Annotate the line with a number
        text(u2_data(end), rf2_data(end), num2str(i), 'FontSize', 6, 'Color', line_color);

        % Add legend entry with escaped underscores
        legend_entries{end+1} = [num2str(i) ' ' strrep(strrep(folder_name, '_results', ''), '_', '\_')];
    catch ME
        % Display a warning and continue with the next iteration
        warning('Could not process folder %s: %s', folder_name, ME.message);
        continue;
    end
end

% Add the legend to the original comparison plot
legend(legend_entries, 'Location', 'eastoutside');

% Save the original comparison plot
savefig(fullfile(compare_folder, 'Force-Displacement_Comparison'));
saveas(gcf, fullfile(compare_folder, 'Force-Displacement_Comparison.png'));
saveas(gcf, fullfile(compare_folder, 'Force-Displacement_Comparison.svg'));

hold off;

% Repeat the try-catch block for the other sections of your MATLAB script
% to ensure the script continues even if an error occurs in one of the sections

