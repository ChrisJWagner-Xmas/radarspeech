clear;clc;

pred_file_path = "C:\\Programming\\GitLab\\radarspeech\\speech_synthesis" ...
                + "\\results\corpus_four_run_1\\log10_mel_power_spectrogram_predicted_2_15_31_56";
pred_files = dir(sprintf("%s\\S00*", pred_file_path));

test_indces = csvread("C:\\Programming\\GitLab\\radarspeech\\speech_synthesis\\results\\test_indices.csv");

index = 1; % test_indces(1);
pred_file_name = sprintf("%s\\%s", pred_files(index).folder, pred_files(index).name);
prediction = csvread(pred_file_name);
[num_frames_pred, num_mels] = size(prediction);

% Find the corresponding ground truth file...
gt_file_path = "C:\\Users\\chris\\Documents\\Institut\\corpora\\speech_synthesis" ...
              + "\\corpus_four_hpc\\S001\vocoder_params_files\\log10_mel_power_spectrogram";
gt_files = dir(sprintf("%s\\S00*", gt_file_path));

for file_index = 1:numel(gt_files)
    file_name = sprintf("%s\\%s", gt_files(file_index).folder,...
                                  gt_files(file_index).name);
    data = csvread(file_name);
    [num_frames_gt, ~] = size(data);
    if(num_frames_gt == num_frames_pred)
        fprintf("Found: %s \n -> %s.\n", pred_files(index).name, gt_files(file_index).name);
        break;
    end
end

gt = csvread(file_name);

subplot(1,2,1);
imagesc(prediction);
subplot(1,2,2);
imagesc(gt);

%% Plot individual mel spectra.
spec_index = 149;
plot(gt(spec_index,:),'-k')
hold on
plot(prediction(spec_index,:),'-b');
hold off