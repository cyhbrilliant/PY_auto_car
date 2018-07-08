import cv2
import numpy as np

def initStereoParam():
    left_camera_matrix = np.array([[240.587683502268, 0.0588924600132440, 164.217295585039],
                                   [0., 240.638124894147, 130.279048832343],
                                   [0., 0., 1.]])
    left_distortion = np.array(
        [[-0.419720106572070, 0.144995517031581, -0.000257788951351011, -0.00112873790658435, 0]])

    right_camera_matrix = np.array([[238.665229474434, -0.0922985912523825, 165.357054341383],
                                    [0., 238.825284504611, 128.508093725883],
                                    [0., 0., 1.]])
    right_distortion = np.array(
        [[-0.418271577008662, 0.141469623697088, 0.000346022599110228, 0.000132022982467251, 0]])

    # om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
    R = np.array([[0.999860733245090, -0.00203853701639580, -0.0165637701451990],
                  [0.00198427687872796, 0.999992613956442, -0.00329160718051818],
                  [0.0165703578675522, 0.00325828166294403, 0.999857393252031]])
    T = np.array([-61.7382355476907, -0.117565994303918, -0.886481060263470])  # 平移关系向量
    size = (320, 240)  # 图像尺寸
    # 进行立体更正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                      right_camera_matrix, right_distortion, size, R,
                                                                      T, alpha=0)
    # 计算更正map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size,
                                                         cv2.CV_16SC2)
    return left_map1, left_map2, right_map1, right_map2, Q

def StereoCompute(frame1, frame2, left_map1, left_map2, right_map1, right_map2, Q):
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    # imgL = img1_rectified
    # imgR = img2_rectified
    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
    # stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=11, P1=8, P2=64)
    # stereo2 = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=15, P1=8, P2=64)
    # print(cv2.StereoSGBM.getMode(stereo))
    disparity = stereo.compute(imgL, imgR)
    # disparity2 = stereo2.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=50, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # disp2 = cv2.normalize(disparity2, disparity2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/10., Q)
    # print(threeD[120, 160, 2])

    return disp, threeD, img1_rectified, img2_rectified
