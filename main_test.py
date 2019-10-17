# # -*- coding: UTF-8 -*-
# import sys, os, dlib, glob, numpy
# from skimage import io
#
# if len(sys.argv) != 5:
#     print('请检查参数是否正确')
#     exit()
# # 1.人脸关键点检测器
# predictor_path = sys.argv[1]
# # 2.人脸识别模型
# face_rec_model_path = sys.argv[2]
# # 3.候选人脸文件夹
# faces_folder_path = sys.argv[3]
# # 4.需识别的人脸
# img_path = sys.argv[4]
#
# # 1.加载正脸检测器
# detector = dlib.get_frontal_face_detector()
# # 2.加载人脸关键点检测器
# sp = dlib.shape_predictor(predictor_path)
# # 3. 加载人脸识别模型
# facerec = dlib.face_recognition_model_v1(face_rec_model_path)
#
# # win = dlib.image_window()
# # 候选人脸描述子list
# descriptors = []
# # 对文件夹下的每一个人脸进行:
# # 1.人脸检测
# # 2.关键点检测
# # 3.描述子提取
#
# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#     print("Processing file: {}".format(f))
#     img = io.imread(f)
#     # win.clear_overlay()
#     # win.set_image(img)
#
#     # 1.人脸检测
#     dets = detector(img, 1)
#     print("Number of faces detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         # 2.关键点检测
#         shape = sp(img, d)
#         # 画出人脸区域和和关键点
#         # win.clear_overlay()
#         # win.add_overlay(d)
#         # win.add_overlay(shape)
#         # 3.描述子提取，128D向量
#         face_descriptor = facerec.compute_face_descriptor(img, shape)
#         # 转换为numpy array
#         v = numpy.array(face_descriptor)
#         descriptors.append(v)
#
# # 对需识别人脸进行同样处理
# # 提取描述子，不再注释
# img = io.imread(img_path)
# dets = detector(img, 1)
# dist = []
# for k, d in enumerate(dets):
#     shape = sp(img, d)
#     face_descriptor = facerec.compute_face_descriptor(img, shape)
#     d_test = numpy.array(face_descriptor)
#
#     # 计算欧式距离
#     for i in descriptors:
#         dist_ = numpy.linalg.norm(i - d_test)
#         dist.append(dist_)
#
# # 候选人名单
# candidate = ['Unknown1', 'Unknown2', 'Shishi', 'Unknown4', 'Bingbing', 'Feifei']
# # 候选人和距离组成一个dict
# c_d = dict(zip(candidate, dist))
# cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
# print('\n The person is: %s' % (cd_sorted[0][0]))
# dlib.hit_enter_to_continue()

## face_detetor.py
import sys
import dlib
import time

time_start = time.process_time()
detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
if len(sys.argv[1:]) > 0:
    img = dlib.load_rgb_image(sys.argv[1])
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))
time_end = time.process_time()
print('totally cost', time_end - time_start)
