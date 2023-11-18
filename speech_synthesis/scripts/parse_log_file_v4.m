function file_content = parse_log_file_v4(full_file_path)
    % Function for parsing the log-files from a full hyperparameter
    % optimization run. 
    % @param full_file_path (str): full file path to the log file.
    
    training_start_token = "started_training";
    test_start_token = "started_testing";
    
    file_content = struct();
    
    % ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    file_dir = dir(full_file_path);
    if(isempty(file_dir))
        error("Could not locate files at %s", full_file_path);
    end
    
    file_name = sprintf("%s\\%s", file_dir(1).folder, file_dir(1).name);
       
    % Read in line by line.
    fid = fopen(file_name);
    
    lines = [];
    line_index = 1;

    while(~feof(fid))
       lines{line_index} = fgetl(fid);
       line_index = line_index + 1;
    end
   
    num_lines = line_index - 1;
    fclose(fid);
       
    % Decode the lines.
    line_index = 1;      
    state = 0; % {parse_train_res, parse_test_res}
    read_values = false;
    epoch = 1;
    
    while(line_index <= num_lines)
        
        current_line = lines{line_index};
        
        switch state                                          
            case 0 % test results.
                if(contains(current_line, training_start_token))
                    state = 1;
                end
            case 1
                if(contains(current_line, test_start_token))
                    state = 2;
                else
                    floats = regexp(current_line,'[+-]?\d+\.?\d*', 'match');
                    file_content.val_error(epoch) = str2num(floats{2});
                    file_content.train_error(epoch) = str2num(floats{3});
                    epoch = epoch + 1;
                end
            case 2
                floats = regexp(current_line,'[+-]?\d+\.?\d*', 'match');
                try
                    file_content.test_error = str2num(floats{1});
                catch
                    file_content.test_error = "n.c."; 
                end
                state = 3;
            case 3
                fprintf("Successfully read file.\n");
        end
        
        line_index = line_index + 1;
    end

    file_content.epochs = epoch-1;
           
end

