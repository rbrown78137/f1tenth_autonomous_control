import cv2 as cv
import random
import numpy as np
import math

# Number of training images = 4 * zooms * vert shifts * horz shifts * rotations

number_of_random_rotations = 1
number_of_random_horizontal_shifts = 2
number_of_random_vertical_shifts = 0
number_of_random_zooms = 0
shift_ratio = 0.5
rotation_max = 60
zoom_max = 0.7
number_of_augmented_images = 4 * (1+number_of_random_horizontal_shifts+number_of_random_vertical_shifts) * (1+number_of_random_rotations) * (1+number_of_random_zooms)


def get_augmented_image_and_ground_truths(image, ground_truth):
    images = []
    images.append(image)
    ground_truths = []
    ground_truths.append(ground_truth)
    # Apply Horizontal Shifts
    for shift_idx in range(number_of_random_horizontal_shifts):
        random_shift = get_random_ratio(shift_ratio)
        images.append(horizontal_shift_reflect(image, random_shift))
        ground_truths.append(horizontal_shift_reflect(ground_truth, random_shift))

    # Apply Vertical Shifts
    for shift_idx in range(number_of_random_vertical_shifts):
        random_shift = get_random_ratio(shift_ratio)
        images.append(vertical_shift_reflect(image, random_shift))
        ground_truths.append(vertical_shift_reflect(ground_truth, random_shift))

    # Flip Horizontally
    image_length_before_step = len(images)
    for image_idx in range(image_length_before_step):
        current_image = images[image_idx]
        current_ground_truth = ground_truths[image_idx]
        images.append(horizontal_flip(current_image))
        ground_truths.append(horizontal_flip(current_ground_truth))

    # Flip Vertically
    image_length_before_step = len(images)
    for image_idx in range(image_length_before_step):
        current_image = images[image_idx]
        current_ground_truth = ground_truths[image_idx]
        images.append(vertical_flip(current_image))
        ground_truths.append(vertical_flip(current_ground_truth))

    # Apply Rotations
    image_length_before_step = len(images)
    for image_idx in range(image_length_before_step):
        current_image = images[image_idx]
        current_ground_truth = ground_truths[image_idx]
        for rotation_idx in range(number_of_random_rotations):
            rotation = get_random_ratio(rotation_max)
            images.append(cropped_rotated_image(current_image, rotation))
            ground_truths.append(cropped_rotated_image(current_ground_truth, rotation))

    # Apply Zooms
    image_length_before_step = len(images)
    for image_idx in range(image_length_before_step):
        current_image = images[image_idx]
        current_ground_truth = ground_truths[image_idx]
        for zoom_idx in range(number_of_random_zooms):
            zoom_value = get_random_zoom(zoom_max)
            images.append(zoom(current_image, zoom_value))
            ground_truths.append(zoom(current_ground_truth, zoom_value))

    return images, ground_truths


def resize_image(img, h, w):
    img = cv.resize(img, (h, w), interpolation=cv.INTER_NEAREST)
    return img


def get_random_ratio(ratio_max):
    return random.uniform(-ratio_max, ratio_max)


def get_random_zoom(zoom_min):
    return random.uniform(zoom_min, 1)


def horizontal_shift_resize(img, ratio=0.0):
    if len(img.shape) == 3:
        h, w = img.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img = img[:, :int(w-to_shift), :]
        if ratio < 0:
            img = img[:, int(-1*to_shift):, :]
        img = resize_image(img, h, w)
        return img
    if len(img.shape) == 2:
        h, w = img.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img = img[:, :int(w-to_shift)]
        if ratio < 0:
            img = img[:, int(-1*to_shift):]
        img = resize_image(img, h, w)
        return img


def horizontal_shift_reflect(img, ratio):
    if len(img.shape) == 3:
        h, w = img.shape[:2]
        to_shift = int(w*ratio)
        if ratio > 0:
            img = img[:, :w-to_shift, :]
            img = cv.copyMakeBorder(img, 0, 0, to_shift, 0, cv.BORDER_REFLECT)
        if ratio < 0:
            img = img[:, -1*to_shift:, :]
            img = cv.copyMakeBorder(img, 0, 0, 0, -1*to_shift, cv.BORDER_REFLECT)
        return img
    if len(img.shape) == 2:
        h, w = img.shape[:2]
        to_shift = int(w*ratio)
        if ratio > 0:
            img = img[:, :w-to_shift]
            img = cv.copyMakeBorder(img, 0, 0, to_shift, 0, cv.BORDER_REFLECT)
        if ratio < 0:
            img = img[:, -1*to_shift:]
            img = cv.copyMakeBorder(img, 0, 0, 0, -1*to_shift, cv.BORDER_REFLECT)
        return img


def vertical_shift_resize(img, ratio=0.0):
    if len(img.shape) == 3:
        h, w = img.shape[:2]
        to_shift = h*ratio
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
        if ratio < 0:
            img = img[int(-1*to_shift):, :, :]
        img = resize_image(img, h, w)
        return img
    if len(img.shape) == 2:
        h, w = img.shape[:2]
        to_shift = h*ratio
        if ratio > 0:
            img = img[:int(h-to_shift), :]
        if ratio < 0:
            img = img[int(-1*to_shift):, :]
        img = resize_image(img, h, w)
        return img


def vertical_shift_reflect(img, ratio=0.0):
    if len(img.shape) == 3:
        h, w = img.shape[:2]
        to_shift = int(h*ratio)
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
            img = cv.copyMakeBorder(img, to_shift, 0, 0, 0, cv.BORDER_REFLECT)
        if ratio < 0:
            img = img[int(-1*to_shift):, :, :]
            img = cv.copyMakeBorder(img, 0, -1*to_shift, 0, 0, cv.BORDER_REFLECT)
        return img
    if len(img.shape) ==2:
        h, w = img.shape[:2]
        to_shift = int(h*ratio)
        if ratio > 0:
            img = img[:int(h-to_shift), :]
            img = cv.copyMakeBorder(img, to_shift, 0, 0, 0, cv.BORDER_REFLECT)
        if ratio < 0:
            img = img[int(-1*to_shift):, :]
            img = cv.copyMakeBorder(img, 0, -1*to_shift, 0, 0, cv.BORDER_REFLECT)
        return img


def zoom(img, value):
    if len(img.shape) == 2:
        h, w = img.shape[:2]
        h_taken = int(value*h)
        w_taken = int(value*w)
        h_start = random.randint(0, h-h_taken)
        w_start = random.randint(0, w-w_taken)
        img = img[h_start:h_start+h_taken, w_start:w_start+w_taken]
        img = resize_image(img, h, w)
        return img
    if len(img.shape) == 3:
        h, w = img.shape[:2]
        h_taken = int(value*h)
        w_taken = int(value*w)
        h_start = random.randint(0, h-h_taken)
        w_start = random.randint(0, w-w_taken)
        img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
        img = resize_image(img, h, w)
        return img


def horizontal_flip(img):
    return cv.flip(img, 1)


def vertical_flip(img):
    return cv.flip(img, 0)


def cropped_rotated_image(image, angle):
    image_height, image_width = image.shape[0:2]
    image_rotated = rotate_image(image, angle)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(angle)
        )
    )
    return image_rotated_cropped


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv.INTER_NEAREST
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]
