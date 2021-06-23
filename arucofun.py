import cv2
cv = cv2
import numpy as np
from types import SimpleNamespace as SN
import time, sys, threading
from utils import *

# some love from https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

marker_type = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

nop = lambda x:x
def camloop(f=nop, threaded=False):
    rh = result_holder = SN()

    rh.result = None
    rh.frame = None
    rh.fresh = False

    def get_frame_if_fresh():
        rf = None

        if rh.fresh:
            rh.fresh = False
            rf = rh.frame

        return rf

    rh.get_frame = get_frame_if_fresh

    def view_update(): # please call from main thread
        frame = get_frame_if_fresh()
        if frame is not None:
            cv2.imshow('cam feed', frame)
            k = cv2.waitKey(1) & 0xff
            if k in [27,ord('q'),ord('f')]:
                exit()
            else:
                return True
        else:
            return False

    rh.update = view_update

    def actual_loop():

        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        # cap.set(cv.CAP_PROP_FRAME_WIDTH,320);
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT,240);
        # cap.set(cv.CAP_PROP_GAIN, 5)

        fpslp = lpn_gen(3, 0.6)
        get_fps_interval = interval_gen()

        lps = [lpn_gen(3, 0.6, integerize=True) for i in range(4)]

        while 1:
            delta = get_fps_interval()
            fps = 1/max(fpslp(delta),1e-3)

            watch = interval_gen(1000)
            timing_string = ts = ''

            ret, frame = cap.read()

            ts+=f'read() {lps[0](watch())} '

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            lines = [] # lines
            dfs = [] # draw functions

            def append_draw_functions(f): dfs.append(f)
            def append_lines(l): lines.append(l)

            result = f(frame, append_lines, append_draw_functions)
            ts+=f'f() {lps[1](watch())} '

            for df in dfs:
                df()

            ts+=f'df() {lps[2](watch())} '

            lines.insert(0, ts)
            lines.insert(0, f'fps{int(fps):3d}')

            for idx, s in enumerate(lines):
                dwstext(frame, s,
                    (2,20*(idx+1)), cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    color=(255,255,255),
                )

            rh.result = result
            rh.frame = frame
            rh.fresh = True

            if not threaded:
                view_update()

    if threaded:
        threading.Thread(target=actual_loop, daemon=True).start()
    else:
        actual_loop()

    return rh

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def draw_with_shadow(f):
    def draw(*a, shadow=True, thickness=2, **k):
        color_specified = k['color']
        del k['color']
        if shadow:
            f(*a, **k, thickness = thickness+1, color = (0,0,0),
                # lineType=cv2.LINE_AA,
                )
        k['color'] = color_specified
        f(*a, **k, thickness = thickness,
            # lineType=cv2.LINE_AA,
            )
    return draw

dwsline = draw_with_shadow(cv2.line)
dwscircle = draw_with_shadow(cv2.circle)
dwstext = draw_with_shadow(cv2.putText)

arucoParams = ac = cv2.aruco.DetectorParameters_create()
ac.errorCorrectionRate = 0.9
# ac.aprilTagMaxLineFitMse = 5.0

class Detection:
    def __init__(self, mid, corners):
        self.corners = corners # of shape [4,2]

        self.iage = 0
        self.marker_id = mid
        self.update()

    def update(self):
        tl,tr,br,bl = corners = self.corners

        # self.tl = tl
        # self.tr = tr
        # self.bl = bl
        # self.br = br

        self.cxy = np.sum(corners, 0) * .25
        self.uxy = np.sum(corners[0:2] - corners[2:4], 0) * .5
        self.size = np.sqrt(
            (np.sum(np.square(tl-br)) + np.sum(np.square(tr-bl)))
        )

def detect(image):
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, marker_type,
        parameters=arucoParams)

    detection_d = {}
    detection_l = []

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))

            detection = Detection(markerID, corners)

            detection_d[markerID] = detection
            detection_l.append(detection)

    return detection_d, detection_l
            # # draw the bounding box of the ArUCo detection
            # line(topLeft, topRight, (0,128,255)) # orangy
            # line(topRight, bottomRight, (0,233,0)) # green
            # line(bottomRight, bottomLeft, (0,0,255)) # red
            # line(bottomLeft, topLeft, (255,128,0)) # cyan

def getgray(frame, downs=1):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    for i in range(downs):
        gray = cv2.pyrDown(gray)
    # gray = cv2.pyrDown(gray)
    return gray

def draw_trackpoints(image, p0, color=(0,0,255), radius=3, thickness=1):
    for k0 in p0:
        dwscircle(image,
            (int(k0[0]), int(k0[1])),
            radius,
            color=color,
            shift=0, shadow=False, thickness=thickness)

def draw_tracks(image, p0, p1):
    for k0, k1 in zip(p0, p1):
        dwsline(image, (int(k0[0]), int(k0[1])),
            (int(k1[0]), int(k1[1])),
            color=(200,200,0), shift=0, shadow=False, thickness=1)

# draw a square
square_lines = np.array([
    [[-1,1],[1,1]],
    [[1,1],[1,-1]],
    [[1,-1],[-1,-1]],
    [[-1,-1],[-1,1]],
], dtype='float32')

square_lines_splitted = np.array([
    [
        [l[0], l[0]+(l[1]-l[0])*0.2], [l[1], l[1] + (l[0] - l[1])*0.2]
    ]
    for l in square_lines
]).reshape((-1, 2, 2))

def apply_transform(points, affine):
    shape = list(points.shape)
    lps = len(shape)-1
    shape[-1]=1
    padshape = tuple(shape)
    pad = np.ones(padshape)
    points = np.concatenate((points, pad), axis=lps, dtype='float32')

    return np.matmul(points, affine)[...,0:2]

def draw_transform_indicator(image, at):
    s = image.shape

    # place it at center of frame
    lines = square_lines_splitted * s[1] * 0.1 + \
        np.array([s[1], s[0]]) * 0.5

    bc = np.array([0,255,0], dtype='float32')

    liness = []
    for i in range(5):
        liness.append(lines)
        # transform the linepoints
        lines = at.transform(lines)

    for i,lines in enumerate(reversed(liness)):
        pwr = 0.8**(len(liness)-i-1)
        c = (bc*pwr).astype('int').tolist()

        for p0,p1 in lines*8.:
            dwsline(image, (int(p0[0]), int(p0[1])),
                (int(p1[0]), int(p1[1])),
                color=c,
                shift=3, shadow=False, thickness=2)

def affine_estimator_gen():
    old_gray = None

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 200,
                           qualityLevel = 0.03,
                           minDistance = 4,
                           blockSize = 3)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (17,17),
                      maxLevel = 10,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.01))

    def affine_estimator(frame, append_lines, append_draw_function):
        nonlocal old_gray
        result = None

        interval = interval_gen()
        t_total = interval_gen()
        ts = timing_string = ''

        gray = getgray(frame, 2) # downscale

        if old_gray is None:
            old_gray = gray
            return frame

        interval()
        p0 = cv.goodFeaturesToTrack(gray, mask = None, **feature_params)
        ts+=f'GFTT {int(interval()*1000)} '

        status_string = ss = ''

        if p0 is not None:
            ss += f'{len(p0)} feats '

            interval()
            p1, st, err = cv.calcOpticalFlowPyrLK(
                old_gray, gray, p0, None, **lk_params)

            ts+=f'OFPLK() {int(interval()*1000)} '

            p0_tracked = p0t = p0[st==1] * 4. # due to previous downscale
            p1_tracked = p1t = p1[st==1] * 4.

            append_draw_function(lambda:draw_tracks(frame, p0t, p1t))
            # ts+=f'draw trks {int(interval()*1000)} '

            ss += f'{len(p0_tracked)} tracked '

            if len(p0_tracked)>6:
                interval()

                at = AffineTransform()
                at.estimate_from(p0t, p1t)

                if at.has_solution():
                    inliers = at.inliers
                    append_draw_function(
                        lambda:draw_trackpoints(
                            frame,
                            p1t[inliers[:,0]==1],
                            color=(0, 255, 255),
                        )
                    )
                    append_draw_function(
                        lambda:draw_trackpoints(
                            frame,
                            p1t[inliers[:,0]==0],
                            color=(0, 0, 255),
                            thickness=2, radius=5,
                        )
                    )

                    ss+=f'(xform solved) {int(interval()*1000)}'

                    append_draw_function(
                        lambda:draw_transform_indicator(frame, at))

                    # print(retval)
                    result = at

                else:
                    ss+=f'(xform not solved) {int(interval()*1000)}'
            else:
                append_draw_function(lambda:draw_trackpoints(frame, p1t, radius=5, thickness=2))

        append_lines(ss)
        append_lines(ts)
        append_lines(f'affine estm {int(t_total()*1000)}')

        old_gray = gray
        return result

    return affine_estimator

# camloop(affine_estimator_gen())

def mark_detections(image, detection_l):
    for v in detection_l:

        cxy,uxy,size = v.cxy, v.uxy, v.size
        uexy = cxy + uxy
        iage = v.iage

        cx,cy,uex,uey = int(cxy[0]), int(cxy[1]), int(uexy[0]), int(uexy[1])

        iagec = int(np.clip(iage*20, 0, 255))

        dwscircle(image, (cx, cy), int(size*0.2),
            color=(200,200,50) if iage else (50,255,255) , shift=0, thickness=2, shadow=False)


        dwsline(image, (cx, cy), (uex, uey),
            color=(0,255-iagec,iagec), shift=0, thickness=2, shadow=False)

        dwstext(image, str(v.marker_id),
            (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color=(255,255,255), shadow=False,
        )

def dwsline2p_white(append_draw_function,frame, p0, p1):
    append_draw_function(lambda:
        dwsline(frame,
            (int(p0[0]*8), int(p0[1]*8)),
            (int(p1[0]*8), int(p1[1]*8)),
            color=(255,255,200), shift=3)
    )

class AffineTransform():
    def __init__(self):
        self.inliers, self.mat = None, None

    # estimate from a bunch of points
    def estimate_from(self, src, dest):
        retval, inliers = cv.estimateAffine2D(
            src, dest, ransacReprojThreshold=5, maxIters=2000, refineIters=20)

        if not len(retval):
            self.inliers = None
            self.mat = None
            self.matinv = None
            return

        retval = np.append(retval, np.array([[0,0,1]]), axis=0).T
        self.mat = retval
        self.matinv = np.linalg.inv(self.mat)
        self.inliers = inliers

    # calculate from 3 points
    def calculate_from(self, src, dest):
        retval = cv.getAffineTransform(src, dst)
        retval = np.append(retval, np.array([[0,0,1]]), axis=0).T
        self.mat = retval

    def transform(self, points, inverse=False):
        # points of shape (n, 2)
        assert points.shape[-1]==2

        # pad points from (n,2) to (n,3) with ones
        shape_list = list(points.shape)
        shape_list[-1]=1 # (n, 1)
        pad = np.ones(shape_list) # ones of shape (n, 1)
        points = np.concatenate((points, pad), axis=-1, dtype='float32')

        # do the transform
        mat = self.mat if not inverse else self.matinv
        return np.matmul(points, mat)[...,0:2]

    def has_solution(self):
        return self.mat is not None

def aruco_tracker_gen():
    ind = {}
    ae = affine_estimator_gen()
    tabletop_square_transform = tst = AffineTransform()

    def aruco_tracker(frame, append_lines, append_draw_function):
        nonlocal ind, ae, tst

        interval = interval_gen()

        transform = ae(frame, append_lines, append_draw_function)

        interval()
        dd, dl = detect(frame)
        append_lines(f'mrkr det in {int(interval()*1000)}')

        if transform is not None:
            to_interp = {}
            for k,v in list(ind.items()): # for each detection in cache
                if k not in dd: # if not detected in this frame
                    v.corners = transform.transform(v.corners)
                    v.update()

                    v.iage+=1
                    if (v.iage>16)\
                        or (v.cxy[0]<0) or (v.cxy[0]>frame.shape[1])\
                        or (v.cxy[1]<0) or (v.cxy[1]>frame.shape[0]):
                        del ind[k]

            ind.update(dd)
        else:
            ind = dd.copy()
        tags = ind

        append_draw_function(lambda:mark_detections(frame, tags.values()))

        # draw line between tag 0-4
        for i in range(4):
            try:
                # print(i in tags, (i+1)%4 in tags)
                p0 = tags[i]
                p1 = tags[(i+1) % 4]
            except:
                continue
            else:
                # print(i, (i+1)%4)
                dwsline2p_white(append_draw_function,frame, p0.cxy, p1.cxy)


        # find transform from tag 0-4 to unit square
        found = [] # sources
        unit_square = [] # targets
        unit_square_coords = usc = {0:[1,1], 1:[1,0], 2:[0,0], 3:[0,1]}

        for i in range(4):
            if i in tags:
                found.append(tags[i].cxy)
                unit_square.append(usc[i])

        if len(found)>=4:
            at = AffineTransform()
            at.estimate_from(np.array(found), np.array(unit_square))
            if at.has_solution():
                tst = at

        # draw the transform found by converting a cross into screen space
        if tst.has_solution():
            append_lines(f'tst has solution')
            cross_lines = np.array([[[.5,0], [.5,1]], [[0,.5], [1,.5]]])
            cross_lines = tst.transform(cross_lines, inverse=True)
            for p0, p1 in cross_lines:
                dwsline2p_white(append_draw_function,frame, p0, p1)
        else:
            append_lines('tst no solution')

        return tags, tst

    return aruco_tracker

if __name__ == '__main__':
    cl = camloop(aruco_tracker_gen(), threaded=True)

    while 1:
        cl.update()
        # print(cl.result)
        time.sleep(0.1)

def denseflow_gen():
    last_gray = None
    flow = None

    def denseflow(frame):
        nonlocal last_gray, flow

        interval = interval_gen()

        gray = getgray(frame, 2)

        if last_gray is None:
            last_gray = gray
            return

        flow = cv.calcOpticalFlowFarneback(
            last_gray, gray,
            flow = flow,
            pyr_scale = .75,
            levels = 15,
            winsize = 15,
            iterations = 15,
            poly_n = 7, poly_sigma=1.5,
            flags = cv.OPTFLOW_USE_INITIAL_FLOW if flow is not None else 0,
        )

        # absflow = np.sqrt(np.sum(np.square(flow), axis=-1)) * 0.02
        absflow = np.zeros(flow.shape[0:2]+(3,), dtype='float32')
        absflow[...,1] = flow[...,0] * 0.01 + 0.5
        absflow[...,2] = flow[...,1] * 0.01 + 0.5

        j = int(interval()*1000)
        print(f'flow calc cost {j}')

        cv2.imshow('flow', absflow)

        last_gray = gray
        return flow

    return denseflow

# camloop(denseflow_gen())
