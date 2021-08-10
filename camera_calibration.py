import smallfix

import cv2
cv = cv2
import numpy as np
from types import SimpleNamespace as SN
import time, sys, threading, math
from utils import *

from shelftest import get_shelf

from functools import lru_cache

data = get_shelf()
print('data:', list(data.keys()))

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

@lru_cache()
def generate_image_grids(h, w):
    ch, cw = h//2, w//2

    k = 80

    sh = ch - ((h-ch) // k +1) * k
    sw = cw - ((w-cw) // k +1) * k

    eh = ch + ((h-ch) // k +1) * k
    ew = cw + ((w-cw) // k +1) * k

    l = []
    for i in range(sw, ew+1, k):
        for j in range(sh, eh+1, k):
            x = i
            y = j

            l.append([[x-k*.5,y], [x,y]])
            l.append([[x,y], [x+k*.5,y]])
            l.append([[x,y], [x,y+k*.5]])
            l.append([[x,y-k*.5], [x,y]])

    return np.array(l, dtype='float32')

lastcm = None
last_tl = None

def draw_grid_distorted(fo, cm, dc, ncm):
    global lastcm, last_tl

    frame = fo.frame

    from arucofun import dwsline

    h, w = frame.shape[0:2]
    l = generate_image_grids(h,w) #(n, 2, 2)

    if cm is not None:
        if (lastcm is not None) and ((cm-lastcm).mean()==0) and (last_tl is not None):
            l = last_tl
        else:
            l = l.reshape(-1, 2)
            nl = cv.undistortPoints(l, cm, dc, P=ncm)
            nl = nl.reshape(-1, 2, 2)
            l = (nl*4).round().astype('int32')

            last_tl = l
            lastcm = cm

            print('undistort grid points')

    else:
        if last_tl is None:
            last_tl = (l*4).round().astype('int32')
            print('mul4')
        l = last_tl

    c = (128,128,128)
    # for p1, p2 in l:
    #     cv2.line(frame, p1, p2,
    #         thickness=1, color=c, shift=2,
    #         lineType=cv2.LINE_AA,
    #         )

    cv2.polylines(frame, l,
        color = c,
        thickness=1,
        isClosed=False, shift=2,
        lineType = cv2.LINE_AA,
    )

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

def chessboard_finder_gen(calibrate=True):
    last = None
    buf = [] # imagePoints buffer
    buf_wp = []
    calibs = None

    cm,dc,ncm = [None]*3

    counter = 0

    m1m2 = None

    with get_shelf() as d:
        d['has_lens_profile'] = True

        if 'cm' in d: cm = d['cm']
        if 'dc' in d: dc = d['dc']
        if 'ncm' in d: ncm = d['ncm']

    def chessboard_finder(frameobj):
        nonlocal last, buf, calibs, cm,ncm,dc,buf_wp,counter,m1m2

        fo = frameobj
        frame = fo.frame

        # bp = cv2.SimpleBlobDetector_Params()
        # bp.minArea=20
        # bp.maxArea=100000
        # blobDetector = cv.SimpleBlobDetector_create(bp)

        # should we look for board(expensive)?
        if calibrate:
            if counter>=0:
                counter = (counter+1) % 10
                look_for_board = counter==0
            else:
                look_for_board = True
        else:
            look_for_board = False

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


                # avgdist
                if len(buf)!=0:
                    avgdist = np.average(np.abs(buf[-1] - centers))
                    if avgdist > 10:
                        # far enough from last
                        avgdist_good = True
                    else:
                        avgdist_good = False
                else:
                    avgdist_good = True

                # estimate shakiness of image
                if last is None:
                    last = centers
                else:
                    if avgdist_good:

                        shakeness = np.average(np.abs(centers - last))
                        if shakeness < .5:
                            should_calib = True
                        else:
                            print('too shaky!', shakeness)
                    else:
                        print(f'(got {len(buf)}) move to next orientation.')

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
                        k = 0.5
                        while 1:
                            ncm = cm.copy()
                            ncm[0:2,0:2] *= k # smaller crop to deal with fisheye
                            h,w = frame.shape[0:2]

                            p0 = [w//2, -h//20] # top side of original image
                            p1 = cv.undistortPoints(np.array([p0],dtype='float32'), cm, dc, P=ncm)[0][0]
                            print(p1.shape)
                            if p1[0]<0 or p1[1]<0:
                                print('k taken as ', k)
                                break
                            else:
                                k+=0.05

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

                    m1m2 = None

        text = ''
        if cm is None or dc is None or ncm is None:
            text+=f'camera not calibrated {int(interval()*1000)}'

        else:
            text+=f'camera calibrated in  {int(interval()*1000)}'

            if m1m2 is None:
                m1,m2 = cv.initUndistortRectifyMap(
                    cameraMatrix=cm,
                    distCoeffs=dc,
                    R=None,
                    newCameraMatrix=ncm,
                    size = frame.shape[0:2][::-1],
                    m1type = cv.CV_16SC2,
                )
                m1m2 = m1,m2
            else:
                m1,m2 = m1m2

            ud = cv.remap(frame, m1, m2,
                interpolation = cv.INTER_CUBIC)
            # ud = cv.undistort(frame, cm, dc, newCameraMatrix=ncm)
            text=f'undistorted in {int(interval()*1000)} '+text

            fo.frame[:] = ud[:] # force write
            # print(distCoeffs)
        fo.drawtext(text)

        def dgd():
            itvl = interval_gen()
            draw_grid_distorted(fo, cm, dc, ncm)
            fo.drawtext(f'grid drawn in {int(lp_t_grid(itvl()*1000))}')

        fo.draw(dgd)

        return None

    lp_t_grid = lpn_gen(3, 0.75)

    return chessboard_finder

if __name__ == '__main__':
    from arucofun import *

    # cl = camloop(chessboard_finder_gen(calibrate=False), threaded=True)
    cl_rh = camloop(chessboard_finder_gen(calibrate=True), threaded=True)

    while 1:
        cl_rh.update()
