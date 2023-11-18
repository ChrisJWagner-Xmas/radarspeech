% Read in the hyperparameters and training/test results from the text
% files.
close all; clc; clear;

addpath("C:\Programming\GitLab\radarspeech\speech_synthesis\scripts");

results_folder_path = "C:\Programming\GitLab\radarspeech\speech_synthesis\" ...
    + "saved_results\corpus_six\crn\phase_mag_db_delta_phase_delta";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

log_file_paths = sprintf("%s\\log*", results_folder_path);
output_folder_paths = sprintf("%s\\run*", results_folder_path);

log_files = dir(log_file_paths);
output_folder = dir(output_folder_paths);
paste_results = zeros(numel(log_files), 3);

for n = 1:numel(log_files)
    % Read the log file.
    file_name = sprintf("%s\\%s", log_files(n).folder, log_files(n).name);
    log_results{n} = parse_log_file_v4(file_name);
    % Get the subject id.
    str = split(file_name,'_');
    str = split(str{end},'.');
    subject_id{n} = str{1};
    % Read the hp file.
    folder_name = sprintf("%s\\%s", output_folder(n).folder, output_folder(n).name);
    file = dir(sprintf("%s\\hyper*", folder_name));
    file_name = sprintf("%s\\%s", file.folder, file.name);
    json_str = fileread(file_name);
    hyperparameters{n} = jsondecode(json_str);
    % Read the io parameters.
    folder_name = sprintf("%s\\%s", output_folder(n).folder, output_folder(n).name);
    file = dir(sprintf("%s\\io*", folder_name));
    file_name = sprintf("%s\\%s", file.folder, file.name);
    json_str = fileread(file_name);
    io_parameters{n} = jsondecode(json_str);
    % Copy-paste results for excel table.
    num_epochs{n} = log_results{n}.epochs - hyperparameters{n}.patience;
    paste_results(1,n) = num_epochs{n};
    paste_results(2,n) = log_results{n}.val_error(end);
    paste_results(3,n) = log_results{n}.train_error(end);
    paste_results(4,n) = log_results{n}.test_error;
end

for n = 1:numel(log_files)
    figure("Name", "Training results")
    plot(1:log_results{n}.epochs, log_results{n}.val_error,'-b');
    fprintf("Number of epochs to train: %d.\n", num_epochs{n});
    title_name = "input params: " + join(io_parameters{n}.transform_keys,",") ...
        + ", " + subject_id{n} + ", epochs: " ...
        + num2str(num_epochs{n});
    title(title_name,'interpreter','none')
    hold on
    plot(1:log_results{n}.epochs, log_results{n}.train_error,'-k')
    legend('Validation error', 'Training error')
    set(gcf,"Position",[172    79   560   420]);
end

%%
o(:,1) = cellfun(@(x) x.lstm_hidden_size, hyperparameters)';
o(:,2) = cellfun(@(x) x.lstm_num_layers, hyperparameters)';
o(:,3) = cellfun(@(x) x.learning_rate, hyperparameters)';
o(:,4) = cellfun(@(x) x.val_error(x.epochs-15), log_results)';

scatter3(o(:,1),o(:,2),o(:,4),'o','MarkerFaceColor','b')

