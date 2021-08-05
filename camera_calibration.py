import cv2
cv = cv2
import numpy as np
from types import SimpleNamespace as SN
import time, sys, threading, math
from utils import *

from shelftest import get_shelf

data = get_shelf()
print('data:', list(data.keys()))

clip = lambda a,b: lambda x: min(b,max(a,x))
clip255 = clip(0,255)

def draw_chessboard_corners(frameobj, corners):
    from arucofun import dwscircle

    fo = frameobj

    for idx, i in enumerate(corners):
        x,y = i
        k = idx*10 % 511

        r,g,b = clip255(k),clip255(511-k), 20
        bgr = b,g,r

        fo.draw(lambda x0=x,y0=y,c=bgr: dwscircle(fo.frame,
            (int(x0), int(y0)),
            radius=3,
            color=c,
            shadow=True, thickness=2))

def draw_grid_distorted(frame, cm, dc, ncm):
    from arucofun import dwsline

    h, w = frame.shape[0:2]
    ch, cw = h//2, w//2

    k = 80

    sh = ch - ((h-ch) // k +1) * k
    sw = cw - ((w-cw) // k +1) * k

    l = []
    for i in range(sw, w, k):
        for j in range(sh, h, k):
            x = i
            y = j

            l.append([[x,y], [x+k,y]])
            l.append([[x,y], [x,y+k]])

            #(n, 2, 2)

    if cm is not None:
        l = np.array(l, dtype='float64')
        l.shape = l.shape[0] * 2, l.shape[2]

        nl = l.copy()
        # # nl3d = cv.convertPointsToHomogeneous(nl)
        # map1, map2 = cv.initUndistortRectifyMap(cm, dc, None, cm,
        #     (w,h), cv.CV_32FC1)
        #
        # nl[:,0] = map1.at(nl[:,0])
        # nl[:,1] = map2.at(nl[:,1])
        #
        # # nldist,jacobian = cv.projectPoints(nl3d, (0,0,0),(0,0,0), cm, dc)
        # # nl = nldist
        nl = cv.undistortPoints(nl, cm, dc, P=ncm)

        nl.shape = nl.shape[0]//2, 2, 2
        l = nl.round().astype('int32')

    for p1, p2 in l:
        dwsline(frame, (p1[0], p1[1]), (p2[0], p2[1]),
            thickness=1,color=(128,128,128), shadow=False)

# https://github.com/Abhijit-2592/camera_calibration_API
def asymmetric_world_points(rows, cols):
    pattern_points = []
    for i in range(rows):
        for j in range(cols):
            x = j/2
            if j%2 == 0:
                y = i
            else:
                y = i + 0.5
            pattern_points.append([x,y,0])
    return pattern_points

def grid_world_points():
    k = []
    for i in range(9):
        for j in range(14):
            k.append([j, i, 0])

    return k

def chessboard_finder_gen():
    last = None
    buf = [] # imagePoints buffer
    buf_wp = []
    calibs = None

    cm,dc,ncm = [None]*3

    counter = 0

    with get_shelf() as d:
        d['has_lens_profile'] = True

        if 'cm' in d: cm = d['cm']
        if 'dc' in d: dc = d['dc']
        if 'ncm' in d: ncm = d['ncm']

    def chessboard_finder(frameobj):
        nonlocal last, buf, calibs, cm,ncm,dc,buf_wp,counter

        fo = frameobj
        frame = fo.frame

        # bp = cv2.SimpleBlobDetector_Params()
        # bp.minArea=20
        # bp.maxArea=100000
        # blobDetector = cv.SimpleBlobDetector_create(bp)

        # should we look for board(expensive)?
        if counter>=0:
            counter = (counter+1) % 10
            look_for_board = counter==0
        else:
            look_for_board = True

        interval = interval_gen()

        interval()

        if look_for_board:
            retval, corners = cv.findChessboardCornersSB(
                frame,
                (14,9),
                # flags = cv.CALIB_CB_LARGER,
            )

            should_calib = False
            # return None # debug

            # board found
            if retval:
                fo.drawtext('chessboard found')
                counter = -1 # from now on look for board in every frame

                centers = corners
                centers = [i[0] for idx, i in enumerate(centers)]

                # for i, row in enumerate(meta):
                #     for j, col in enumerate(row):
                #         k = centers[i*14 + j]
                #         x,y = k
                #         if col != 0:
                #             fo.draw(lambda x0=x,y0=y: dwscircle(frame,
                #                 (int(x0), int(y0)),
                #                 radius=2,
                #                 color=(255,128,255),
                #                 shadow=True, thickness=1))

                draw_chessboard_corners(frameobj, centers)

                centers = np.array(centers, dtype='float32')


                # estimate shakiness of image
                if last is None:
                    last = centers
                else:
                    shakeness = np.average(np.abs(centers - last))
                    if shakeness < .5:
                        # stable

                        if len(buf)!=0:
                            avgdist = np.average(np.abs(buf[-1] - centers))
                            if avgdist > 10:
                                # far enough from last
                                should_calib = True
                        else:
                            should_calib = True
                    else:
                        print('shakiness', shakeness)

                    last = centers

            if should_calib:
                buf.append(np.array(centers, 'float32'))
                buf_wp.append(np.array(grid_world_points(), 'float32'))
                print('buflen',len(buf))


                lbuf = len(buf)
                retval = None
                if lbuf >= 1: # enough data

                    # pwc = asymmetric_world_points(4, 11)
                    # pwc = grid_world_points()

                    print('cal with', lbuf, 'frames')

                    # print(pwc.shape)
                    # objectPoints = np.array([pwc]*lbuf, dtype='float32')
                    objectPoints = buf_wp

                    # must be float32
                    # https://stackoverflow.com/a/58116723

                    # print(objectPoints.shape)

                    # imagePoints = np.array(buf, dtype='float32')
                    imagePoints = buf
                    # print(imagePoints.shape)

                    retval, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints, stdDeviationsIntrinsics, stdDeviationsExtrinsics, stdDeviationsObjPoints, perViewErrors = cv.calibrateCameraROExtended(
                    # retval, cameraMatrix, distCoeffs, rvecs, tvecs,\
                    # stdDeviationsIntrinsics, stdDeviationsExtrinsics,\
                    # perViewErrors = \

                        objectPoints = objectPoints,
                        # objectPoints = cv.CALIB_CB_ASYMMETRIC_GRID,
                        imagePoints = imagePoints,
                        imageSize = frame.shape[0:2][::-1],
                        cameraMatrix = np.zeros((3,3)),
                        distCoeffs = np.zeros((8,)),

                        iFixedPoint = 0,

                        flags= 0
                            # |cv.CALIB_RATIONAL_MODEL
                            # |cv.CALIB_THIN_PRISM_MODEL
                            |cv.CALIB_TILTED_MODEL
                    )

                if retval:
                    # successfully calibrated
                    # calibs = cameraMatrix, distCoeffs
                    # cm,dc = calibs
                    cm,dc = cameraMatrix, distCoeffs

                    if 0:
                        retval, validpixroi = cv.getOptimalNewCameraMatrix(
                            cm, dc, frame.shape[0:2], alpha=0,
                            centerPrincipalPoint=False)

                        ncm = retval
                    else:
                        ncm = cm.copy()
                        ncm[0:2,0:2] *= 0.8 # smaller crop to deal with fisheye

                    print('dist coeffs')

                    coefnames = 'k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]'.replace('[', '').replace(']','').split(',')

                    for name, coef in zip(coefnames, distCoeffs):
                        print(name, coef)

                    stddevnames = 'fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,τx,τy'.split(',')

                    print('std intrin')
                    for name, std in zip(stddevnames, stdDeviationsIntrinsics):
                        print(name, std)

                    print('write to db')
                    with get_shelf() as d:
                        d['dc'] = dc
                        d['cm'] = cm
                        d['ncm'] = ncm


        if cm is None or dc is None or ncm is None:
            fo.drawtext(f'camera not calibrated {int(interval()*1000)}')

        else:
            fo.drawtext(f'camera calibrated in  {int(interval()*1000)}')

            ud = cv.undistort(frame, cm, dc, newCameraMatrix=ncm)
            fo.frame[:] = ud[:] # force write
            # print(distCoeffs)

        fo.draw(lambda:draw_grid_distorted(frame, cm, dc, ncm))

        return None

    return chessboard_finder

if __name__ == '__main__':
    from arucofun import *

    cl = camloop(chessboard_finder_gen(), threaded=True)

    while 1:
        cl.update()
        time.sleep(0.1)
