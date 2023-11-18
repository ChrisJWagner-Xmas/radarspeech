function distance_dB = calc_spectral_distortion(P,P_hat,reduction)
    % Calculates the mel-cepstral distortion between two mel-spectra
    % according to "MAPPING FROM ARTICULATORY MOVEMENTS TO VOCAL TRACT SPECTRUM WITH
    % GAUSSIAN MIXTURE MODEL FOR ARTICULATORY SPEECH SYNTHESIS, 2004"
    % Also the same as in https://github.com/MattShannon/mcd/blob/master/mcd/metrics.py
    
    % If P and P_hat are spectrograms, each spectrum is a ROW vector.
    % @param P: input power spectrum or linear spectrogram.
    % @param P: input power spectrum or linear spectrogram for comparison (e.g.,
    %           a predicted power spectrum or spectrogram).
    % @param reduction: flag to return a scalar value or the frame-wise
    %                   LSP as a vector. Can be 'mean', 'sum', or 'none'
    
    if(~strcmp(reduction,'none') && ...
        ~strcmp(reduction,'mean') && ...
        ~strcmp(reduction,'sum'))
        error("reduction can only take the values 'none', 'mean' and 'sum'");
    end
    
    distance_dB = [];
    
    [P_dim1, P_dim2] = size(P); % [num_frames, num_frequencies]
    [P_hat_dim1, P_hat_dim2] = size(P_hat); % -"-
    
    if(P_dim1 ~= P_hat_dim1)
        error("Error: first dimension of input P (%d) and P_hat (%d) are not equal.\n", ...
            P_dim1, P_hat_dim1);
    end
    if(P_dim2 ~= P_hat_dim2)
        error("Error: second dimension of input P (%d) and P_hat (%d) are not equal.\n", ...
            P_dim2, P_hat_dim2);
    end
    
    distance_dB = zeros(1,P_dim1);
    
    log_factor = (10/log(10))*sqrt(2);
    
    for frame_index = 1:P_dim1
        spectral_difference = P(frame_index,:) - P_hat(frame_index,:);
        spectral_difference = reshape(spectral_difference,1,[]); % ensure row vector.
        distance_dB(frame_index) = log_factor*sqrt(spectral_difference*spectral_difference');    
    end
    
    % Calculate the sum if specified.
    if(strcmp(reduction,'sum'))
        distance_dB = sum(distance_dB);
    end
    
    % Calculate the mean if specified.
    if(strcmp(reduction,'mean'))
        distance_dB = sum(distance_dB)/P_dim1;
    end

end

