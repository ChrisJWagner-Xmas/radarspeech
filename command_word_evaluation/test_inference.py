import torch
from datasets.import_rs_data import import_rs_binary_file
from train_model.train_and_evaluate import calculate_delta_sequence
import matplotlib.pyplot as plt
import math
import numpy as np

model_file_path = "checkpoints\\lstm_optim_model_18_22_12.pt"
test_file_path = "C:\\Users\\chris\\Documents\\Institut\\corpora\\command_word_recognition_monopole_2\\corpus\\S001\\SES01\\radar_files\\S001_SES01_CL002_REP001.bin"

# Reload the model as a sanity check and test on the training corpus (in-sample).
loaded_model = torch.load(model_file_path)
radar_data = import_rs_binary_file(test_file_path)

sequence = radar_data.radargrams[1]
sequence_length = torch.unsqueeze(torch.tensor(sequence.shape[0],dtype=torch.int64),dim=0)

# Transform the sequence.
freq_start_index = 0
freq_stop_index = 85
sequence = torch.from_numpy(sequence[:,freq_start_index:freq_stop_index])
delta_sequence = calculate_delta_sequence(sequence)
magnitude = torch.abs(sequence)
magnitude = magnitude/torch.max(magnitude)
delta_mag = torch.abs(delta_sequence)
delta_mag = delta_mag/torch.max(delta_mag)
delta_phase = torch.angle(delta_sequence)/math.pi

# Concatenate the features and add the batch dimension to the tensor.
features = torch.unsqueeze(torch.cat((magnitude, delta_mag, delta_phase), dim=1), dim=0)

np.savetxt("flattened_features.csv", torch.flatten(features).numpy())
plt.imshow(features[0])
plt.show()

predictions = loaded_model(features, sequence_length)
plt.plot(torch.exp(predictions).detach().numpy().T)
plt.show()

y_pred = torch.argmax(predictions, dim=1)

print("Predicted class %d" % y_pred.item())
