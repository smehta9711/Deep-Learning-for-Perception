import pixellib
from pixellib.instance import instance_segmentation
import cv2


segmentation_model = instance_segmentation()
segmentation_model.load_model('mask_rcnn_coco.h5')

frame = cv2.imread('mask_948.png')
res = segmentation_model.segmentFrame(frame, show_bboxes=True)
image = res[1]

cv2.imwrite('maskkk.png',image)
cv2.imshow('Instance Segmentation', image)


cv2.waitKey(0)
cv2.destroyAllWindows()