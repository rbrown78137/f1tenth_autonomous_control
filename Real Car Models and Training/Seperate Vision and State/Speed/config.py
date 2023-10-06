input_image_width = 256
input_image_height = 256
input_channels = 1

dataset_path = "/home/ryan/TrainingData/simulator_speed_data"
dataset_image_folder_name = "photos"
dataset_input_file = "data.bin"

input_byte_size = 8
input_file_type = 'd'
byte_offset_between_indices = 0

batch_size = 16
number_of_epochs = 500
learning_rate = 0.0001

# Used to Find % of train / test values that fall within a certain distance of the true value
# (this distance being the square root of the values provided below)
loss_cutoff_for_accuracy = [1, 0.5, 0.25, 0.1]
