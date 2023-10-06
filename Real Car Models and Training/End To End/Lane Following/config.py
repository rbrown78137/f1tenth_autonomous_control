input_image_width = 256
input_image_height = 256
number_of_image_classifications = 4
input_dimensions = 3

dataset_path = "/home/ryan/Real Car Data Sets/Recording 5"
dataset_image_folder_name = "real_car_photos"
dataset_input_file = "steeringData.bin"

input_byte_size = 8
input_file_type = 'd'

batch_size = 32
number_of_epochs = 500
learning_rate = 0.00001

# Used to Find % of train / test values that fall within a certain distance of the true value
# (this distance being the square root of the values provided below)
loss_cutoff_for_accuracy = [0.03046176808, 0.01, 0.00761544202, 0.0025]
