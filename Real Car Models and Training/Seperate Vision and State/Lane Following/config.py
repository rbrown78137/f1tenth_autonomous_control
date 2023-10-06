input_image_width = 256
input_image_height = 256
input_channels = 1

dataset_path = "/home/ryan/TrainingData/simulator_following_data"
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
loss_cutoff_for_accuracy = [0.174533, 0.0872665, 0.0174533, 0.004363323]
# 10, 5, 1, 0.25 DEGREES