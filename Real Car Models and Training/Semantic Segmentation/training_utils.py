import torch
import matplotlib.pyplot as plt
import time
import pylab

import config


def display_prediction_and_distribution(model,test_images):
    sample_1_expected_value, sample_1_variance = model.get_distribution(test_images[0].unsqueeze(0))
    sample_2_expected_value, sample_2_variance = model.get_distribution(test_images[1].unsqueeze(0))
    prediction_1 = sample_1_expected_value.argmax(0)
    prediction_2 = sample_2_expected_value.argmax(0)
    confidence_tensor_1 = torch.zeros(prediction_1.shape)
    for row in range(confidence_tensor_1.shape[0]):
        for col in range(confidence_tensor_1.shape[1]):
            confidence_tensor_1[row][col] = sample_1_variance[prediction_1[row][col]][row][col]
    confidence_tensor_2 = torch.zeros(prediction_2.shape)
    for row in range(confidence_tensor_2.shape[0]):
        for col in range(confidence_tensor_2.shape[1]):
            confidence_tensor_2[row][col] = sample_2_variance[prediction_2[row][col]][row][col]

    f = plt.figure()
    # Image 1
    time.sleep(0.25)
    f.add_subplot(2, 3, 1)
    plt.imshow(prediction_1)
    f.add_subplot(2, 3, 2)
    plt.imshow(confidence_tensor_1)
    f.add_subplot(2, 3, 3)
    plt.imshow(test_images[0].permute(1, 2, 0).mul(1 / 256).to("cpu"))

    # Image 2
    f.add_subplot(2, 3, 4)
    plt.imshow(prediction_2)
    f.add_subplot(2, 3, 5)
    plt.imshow(confidence_tensor_2)
    f.add_subplot(2, 3, 6)
    plt.imshow(test_images[1].permute(1, 2, 0).mul(1 / 256).to("cpu"))

    pylab.show()
    time.sleep(0.25)


def display_image(images):
    plt.imshow(images[0].permute(1, 2, 0))


def average_class_iou(prediction_prob, ground_truth):
    prediction = prediction_prob.argmax(1)
    total_iou = 0
    counted_classes = 0
    for x in range(config.number_of_classifications):
        intersection = torch.where(torch.bitwise_and(prediction == x, ground_truth == x), 1, 0)
        union = torch.where(torch.bitwise_or(prediction == x, ground_truth == x), 1, 0)
        if torch.count_nonzero(union) > 0:
            iou = torch.count_nonzero(intersection) / torch.count_nonzero(union)
            total_iou = total_iou + iou
            counted_classes = counted_classes + 1

    average_iou = total_iou / counted_classes
    return average_iou

def class_ious(prediction_prob, ground_truth):
    prediction = prediction_prob.argmax(1)
    total_iou = torch.tensor([0.0]*4)
    counted_classes = torch.tensor([0.0]*4)

    for x in range(config.number_of_classifications):
        intersection = torch.where(torch.bitwise_and(prediction == x, ground_truth == x), 1, 0)
        union = torch.where(torch.bitwise_or(prediction == x, ground_truth == x), 1, 0)
        if torch.count_nonzero(union) > 0:
            iou = torch.count_nonzero(intersection) / torch.count_nonzero(union)
            iou = iou.item()
            total_iou[x] = total_iou[x] + iou
            counted_classes[x] = counted_classes[x] + 1
    return total_iou,counted_classes