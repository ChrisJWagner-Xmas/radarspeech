% Compare the recorded ground truth mel spectrograms to the predicted ones.
clear;clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Important hyperparameters for spectral comparison.
time_context_frames = 50;

do_remove_time_context = true;
plot_results = true;

% Path to ground truth.
% CURRENTLY STORED ON EXTERNAL HARD DRIVE!
corpus_path = "C:\Users\chris\Documents\Institut\corpora\" ...
            + "speech_synthesis\corpus_six\S002\"  ...
            + "input_params_files\tacotron_mel_power_spectrogram";

% Path to predictions.
predictions_path = "C:\Programming\GitLab\radarspeech\speech_synthesis\saved_results\corpus_six\lstm\" ...
    + "phase_mag_db_delta_phase_delta\run_1_41_32_0_S002";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mels_dir = dir(sprintf("%s\\tacotron*", predictions_path));
mels_folder_name = sprintf("%s\\%s", mels_dir.folder, mels_dir.name);
test_indices_file_name = sprintf("%s\\test_indices.txt", predictions_path);

% Load the test indices.
test_indices = readmatrix(test_indices_file_name);
test_indices = reshape(test_indices, [], 1);
test_indices(isnan(test_indices)) = [];
test_indices = sort(test_indices + 1);

gt_dir = dir(sprintf("%s\\*.csv", corpus_path));
if(~isempty(gt_dir))
    gt_dir = gt_dir(test_indices);
else
    fprintf("Could not load any ground truth files from \n %s.\n", corpus_path)
end
pred_dir = dir(sprintf("%s\\*.csv", mels_folder_name));


for pred_index = 1:numel(test_indices)
    fprintf("At file %d\n", pred_index);
    
    if(~isempty(gt_dir))
        gt_file_name = sprintf("%s\\%s", gt_dir(pred_index).folder, gt_dir(pred_index).name);
        gt = readmatrix(gt_file_name);
    end
    pred_file_name = sprintf("%s\\%s", pred_dir(pred_index).folder, pred_dir(pred_index).name);
    pred = readmatrix(pred_file_name);
    
    if(do_remove_time_context && ~isempty(gt_dir))
       gt = gt(time_context_frames:end,:);
       if(size(gt,1) < size(pred,1))
           pred = pred(time_context_frames:end,:);
       end
    end
    
    if(~isempty(gt_dir))
        spec_dist(pred_index,1) = calc_spectral_distortion(10.^gt, ...
                                           10.^pred, 'mean');
    else
        spec_dist(pred_index,1) = -inf;
    end
%     
    if(plot_results)
        subplot(2,1,1);
        if(~isempty(gt_dir))
            imagesc(gt');
        end
        subplot(2,1,2);
        imagesc(pred');
        title(sprintf("Predictions, MCD: %1.3f", spec_dist(pred_index)));
        pause()
    end
end