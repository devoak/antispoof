import cv2



for index in range(1, 10001):
    img = cv2.imread('./datasets/trainSpoofOld/spoof/trainSpoof' + str(index) + '.png')
    height, width = img.shape[:2]
    max_height = 224
    max_width = 224
    if max_height != height or max_width != width:
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imwrite('./datasets/trainSpoof/spoof/trainSpoof' + str(index) + '.png', img)
    print("done", index)


cv2.destroyAllWindows()
