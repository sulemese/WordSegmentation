import cv2
import numpy as np
import matplotlib.pyplot as plt

def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh

def dilate_image(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated

def get_word_contours(image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_bounding_boxes(image, contours, color=(40, 100, 250)):
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    return image

def find_words(contours, dilation_image, min_area=400):
    words_list = []
    for line in contours:
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilation_image[y:y+w, x:x+w]
        cnt, _ = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])
        for word in sorted_contour_words:
            if cv2.contourArea(word) < min_area:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(word)
            words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
    return words_list

def draw_word_boxes(image, words_list, color=(255, 255, 100)):
    merged_boxes = []
    for word_box in words_list:
        x1, y1, x2, y2 = word_box
        merged = False
        for merged_box in merged_boxes:
            if x1 >= merged_box[0] and y1 >= merged_box[1] and x2 <= merged_box[2] and y2 <= merged_box[3]:
                merged = True
                break
            elif x1 <= merged_box[0] and y1 <= merged_box[1] and x2 >= merged_box[2] and y2 >= merged_box[3]:
                merged_boxes.remove(merged_box)
                break
        if not merged:
            merged_boxes.append(word_box)

    for word_box in merged_boxes:
        x1, y1, x2, y2 = word_box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image

if __name__ == "__main__":
    img = cv2.imread("E:\kitap-alintilari_1477811.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Thresholding
    thresh_img = thresholding(img)

    # Dilation
    dilated = dilate_image(thresh_img, (3, 85))

    # Find contours for lines
    contours_lines = get_word_contours(dilated)
    sorted_contours_lines = sorted(contours_lines, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Dilation for words
    dilated2 = dilate_image(thresh_img, (3, 15))

    # Find words
    words_list = find_words(sorted_contours_lines, dilated2)

    # Draw bounding boxes for words
    img_with_words = draw_word_boxes(img.copy(), words_list)

    # Show the result
    plt.imshow(img_with_words)
    plt.show()
