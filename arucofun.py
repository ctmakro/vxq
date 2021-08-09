import smallfix

import cv2
cv = cv2
import numpy as np
from types import SimpleNamespace as SN
import time, sys, threading, math
from utils import *

# some love from https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

marker_type = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

nop = lambda x:x
def camloop(f=nop, threaded=False):

    window_name = 'camfeed'

    rh = result_holder = SN()

    rh.result = None
    # rh.frame = None
    rh.framebuffer = MailWaiter()
    rh.resultbuffer = MailWaiter()

    def view_update(): # please call from main thread
        fresh = False

        if rh.framebuffer.gotmail():
            frameobj = rh.framebuffer.recv()
            cv2.imshow(window_name, frameobj.frame)
            fresh = True
        else:
            k = cv2.waitKey(16) & 0xff
            if k in [27,ord('q'),ord('f')]:
                exit()

        return fresh

    rh.update = view_update

    # def view_update_f(f):
    #     frame = get_frame_if_fresh()
    #     if frame is not None:
    #         f(frame)
    #     else:
    #         return False
    #
    # rh.update_f = view_update_f

    def actual_loop():
        def try_open_camera(id):
            if sys.platform == 'win32':
                cap = cv2.VideoCapture(id, cv2.CAP_MSMF)
            else:
                cap = cv2.VideoCapture(id, cv2.CAP_ANY)

            if not cap.isOpened():
                print("Cannot open camera", id)
                return False

            target_camera = 'ov2659' or 'anc' or 'pro'

            print('target_camera', target_camera)
            if target_camera=='ov2659':
                # cap.set(3,1280); cap.set(4,1024)
                cap.set(3,1600); cap.set(4,1200)
                cap.set(cv.CAP_PROP_FPS, 30)
                # cap.set(cv.CAP_PROP_CONTRAST, -4)
                cap.set(cv.CAP_PROP_EXPOSURE, -10,)
                cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 20)

            elif target_camera == 'anc':
                pass

            elif target_camera == 'pro':
                cap.set(3,1280)
                cap.set(4,720)

                cap.set(cv.CAP_PROP_FPS, 12)

                # cv.CAP_PROP_EXPOSURE, -5,
                cap.set(cv.CAP_PROP_CONTRAST, 0)
                cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)

            print('trying to get h,w')
            height = cap.get(4)
            width = cap.get(3)
            be = cap.getBackendName()

            print('height:', height, 'width:', width, 'backend:', be)

            return cap
            # raise Exception('frame height not 480')

        cap = try_open_camera(1)
        if not cap:
            cap = try_open_camera(0)
        if not cap:
            raise Exception('cannot open any of the cameras')

        # cap.set(cv.CAP_PROP_FRAME_WIDTH,320);
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT,240);
        # cap.set(cv.CAP_PROP_GAIN, 5)

        fpslp = lpn_gen(3, 0.6)
        get_fps_interval = interval_gen()

        lps = [lpn_gen(3, 0.6, integerize=True) for i in range(6)]

        fail_counter = 0
        framebuffer = MailWaiter()
        # flock = threading.Lock()
        # fcond = threading.Condition()
        # framefresh = False

        t_read = 0
        fps = 0

        # read frames from camera nonstop
        def loop_reader():
            nonlocal fail_counter, fps, t_read, framebuffer

            target_fps = 12 # limit framerate
            freq_reg = frequency_regulator_gen(12)

            while 1:
                delta = get_fps_interval()
                fps = 1/max(fpslp(delta),1e-3)

                freq_reg()

                timer = interval_gen(1000)
                ret, frame = cap.read()
                t_read = timer()


                if not ret:
                    fail_counter+=1
                    print(f"Can't receive frame (stream end?) {fail_counter}")
                    if fail_counter<20:
                        time.sleep(.5)
                        continue
                    else:
                        print('too many tries, exiting...')
                        break

                else:
                    # got new frame from camera
                    framebuffer.send(frame)

            exit()

        # threaded read to maximize fps
        threading.Thread(target=loop_reader, daemon=True).start()
        t_drawtext = 0
        t_total = 0

        while 1:
            frame = framebuffer.recv()

            watch_tot = interval_gen(1000)

            timing_string = ts = ''

            watch2 = interval_gen(1000)

            h,w = frame.shape[0:2]

            # smaller image for ease of processing
            optimal_width = ow = 960
            optimal_height = oh = 540

            scale = min(ow/w, oh/h)
            if scale < .9:
                nw = int(w*scale)
                nh = int(h*scale)

                frame = cv2.resize(frame, (nw,nh),
                    # interpolation=cv2.INTER_NEAREST)
                    # interpolation=cv2.INTER_LINEAR)
                    interpolation=cv2.INTER_CUBIC)
                # frame = cv2.pyrDown(frame)

            downsampling_t = watch2()

            lines = [] # text lines to draw
            dfs = [] # draw functions

            frameobj = SN()
            frameobj.frame = frame
            frameobj.draw = lambda f:dfs.append(f)
            frameobj.drawtext = lambda l: lines.append(l)

            watch = interval_gen(1000)
            result = f(frameobj)
            rh.resultbuffer.send(result)
            ts+=f'read() {t_read} f() {lps[1](watch())} '

            for df in dfs:
                df()

            ts+=f'df() {lps[2](watch())} drawtext() {lps[3](t_drawtext)} '
            ts+=f'downsample {downsampling_t}'

            lines.insert(0, f'dnsmpl()+f()+df()+dt() {lps[4](t_total)}')
            lines.insert(0, ts)
            lines.insert(0, f'{frame.shape[0:2]} fps{int(fps):3d}')

            watch3 = interval_gen(1000)

            for idx, s in enumerate(lines):
                dwstext(frame, s,
                    (2,12*(idx+1)), cv2.FONT_HERSHEY_DUPLEX,
                    0.35,
                    color=(0,0,0),
                    thickness=3,
                    shadow=False,
                )
            for idx, s in enumerate(lines):
                dwstext(frame, s,
                    (2,12*(idx+1)), cv2.FONT_HERSHEY_DUPLEX,
                    0.35,
                    color=(255,255,255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                    shadow=False,
                )

            t_drawtext = watch3()
            t_total = watch_tot()

            rh.result = result
            # rh.frame = frame
            # rh.frameobj = frameobj
            # rh.fresh = True

            rh.framebuffer.send(frameobj)

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
ac.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
ac.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG
ac.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR

ac.minMarkerPerimeterRate = 0.02
ac.maxMarkerPerimeterRate = 0.3

ac.polygonalApproxAccuracyRate = 0.08
# ac.polygonalApproxAccuracyRate = 0.04
# ac.maxErroneousBitsInBorderRate = 0.6
# ac.errorCorrectionRate = 0.9
# ac.aprilTagMaxLineFitMse = 5.0

class Detection:
    def __init__(self, mid, corners):
        self.corners = corners # of shape [4,2]

        self.iage = 0
        self.marker_id = mid
        self.update()

    def update(self):
        if not hasattr(self, 'last_corners'):
            self.corners_lp = lpn_gen(3, 0.5)
        corners = self.corners_lp(self.corners)
        corners = self.corners

        tl,tr,br,bl = corners

        # self.tl = tl
        # self.tr = tr
        # self.bl = bl
        # self.br = br

        self.cxy = np.sum(corners, 0) * .25
        self.uxy = np.sum(corners[0:2] - corners[2:4], 0) * .5
        self.rxy = np.sum(corners[1:2:4] - corners[0:2:3], 0) * 1
        self.rexy = self.cxy + self.rxy * 1.75
        # self.rexy = self.cxy + self.rxy * 0
        self.uexy = self.cxy + self.uxy * 1

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
    l = (np.stack([p0,p1], axis=1)*4).astype('int32')

    cv2.polylines(image, l,
        color = (200,200,0),
        thickness=1,
        isClosed=False, shift=2,
        lineType = cv2.LINE_AA,
    )

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

        if not np.isnan(np.sum(lines)):
            for p0,p1 in lines*8.:

                dwsline(image, (int(p0[0]), int(p0[1])),
                    (int(p1[0]), int(p1[1])),
                    color=c,
                    shift=3, shadow=False, thickness=2)

def affine_estimator_gen(skip = False):
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

    # camera distortion
    from camera_calibration import chessboard_finder_gen
    chessboard_finder = chessboard_finder_gen(calibrate=False)

    points_buffer = pb = []

    def affine_estimator(fo):
        nonlocal old_gray, pb

        # deal with camera calibration/ undistortion
        chessboard_finder(fo)
        if skip:
            return None

        frame = fo.frame

        result = None

        interval = interval_gen()
        t_total = interval_gen()
        ts = timing_string = ''

        gray = getgray(frame, 2) # downscale

        if old_gray is None:
            old_gray = gray
            return None

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

            pb.append((p0t, p1t))
            if len(pb)>5:
                pb.pop(0)

            for p0t,p1t in pb:
                fo.draw(lambda p0=p0t, p1=p1t:draw_tracks(frame, p0, p1))

            ts+=f'draw trks {int(interval()*1000)} '

            ss += f'{len(p0_tracked)} tracked '

            if len(p0_tracked)>6:
                interval()

                at = PerspectiveTransform()
                # at = AffineTransform()
                at.estimate_from(p0t, p1t, maxIters=100)

                if at.has_solution():
                    inliers = at.inliers
                    fo.draw(
                        lambda:draw_trackpoints(
                            frame,
                            p1t[inliers[:,0]==1],
                            color=(0, 255, 255),
                        )
                    )
                    fo.draw(
                        lambda:draw_trackpoints(
                            frame,
                            p1t[inliers[:,0]==0],
                            color=(0, 0, 255),
                            thickness=2, radius=5,
                        )
                    )

                    ss+=f'(xform solved) {int(interval()*1000)}'

                    fo.draw(
                        lambda:draw_transform_indicator(frame, at))

                    # print(retval)
                    result = at

                else:
                    ss+=f'(xform not solved) {int(interval()*1000)}'
            else:
                fo.draw(lambda:draw_trackpoints(frame, p1t, radius=5, thickness=2))

        [fo.drawtext(k) for k in
            [ss, ts, f'persp estm {int(t_total()*1000)}']]

        old_gray = gray
        return result

    return affine_estimator

# camloop(affine_estimator_gen())
def mark_detections(image, detection_l):
    for v in detection_l:

        cxy,uxy,rxy,size = v.cxy, v.uxy, v.rxy, v.size
        uexy = v.uexy
        # rexy = cxy + rxy*1.75
        rexy = v.rexy
        iage = v.iage

        cx,cy,uex,uey,rex,rey= round(cxy[0]), round(cxy[1]), \
            round(uexy[0]), round(uexy[1]),\
            round(rexy[0]), round(rexy[1])

        iagec = round(np.clip(iage*20, 0, 255))

        dwscircle(image, (cx, cy), round(size*0.2),
            color=(200,200,50) if iage else (50,255,255) , shift=0, thickness=2, shadow=False)

        redgreen = (0,255-iagec,iagec)
        dwsline(image, (cx, cy), (uex, uey),
            color=redgreen, shift=0, thickness=2, shadow=False)

        # small cross
        cl = round(.3*size)
        dwsline(image, (rex, rey+cl), (rex, rey-cl),
            color=redgreen, shift=0, thickness=1, shadow=False)
        dwsline(image, (rex+cl, rey), (rex-cl, rey),
            color=redgreen, shift=0, thickness=1, shadow=False)

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
    def estimate_from(self, src, dest, **kv):
        params = dict(method=cv.RANSAC and cv.RHO, ransacReprojThreshold=2, maxIters=2000, )
        params.update(kv)


        # retval, inliers = cv.findHomography(
        retval, inliers = cv.estimateAffine2D(
            src, dest, **params)

        if retval is not None and not len(retval):
            self.inliers = None
            self.mat = None
            self.matinv = None
            return

        # print(retval.shape)
        retval = np.append(retval, np.array([[0,0,1]]), axis=0).T
        self.mat = retval
        # self.mat = retval.T
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

class PerspectiveTransform(AffineTransform):
    # estimate from a bunch of points
    def estimate_from(self, src, dest, **kv):
        params = dict(method=cv.RANSAC and cv.RHO, ransacReprojThreshold=2)
        params.update(kv)

        retval,inliers = cv.findHomography(
        # retval, inliers = cv.estimateAffine2D(
            src, dest, **params)

        if retval is None or not len(retval):
            self.inliers = None
            self.mat = None
            self.matinv = None
            return

        self.mat = retval.T
        self.matinv = np.linalg.inv(self.mat)
        self.inliers = inliers

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
        result = np.matmul(points, mat)

        xy = result[...,0:2]
        w = result[...,2:3]
        w[w==0] = np.inf
        return xy/w


unit_square_coords = {0:[1,1], 1:[1,0], 2:[0,0], 3:[0,1]}
unit_square_coords = [[1,1], [1,0], [0,0], [0,1]]

def aruco_tracker_gen():
    ind = {}
    ae = affine_estimator_gen(skip=True)
    err = 0
    frametimer = interval_gen()

    def aruco_tracker(fo):
        nonlocal ind, ae, err

        frame = fo.frame

        interval = interval_gen()

        transform = ae(fo)

        interval()

        err+=frametimer()
        if err>=0.25:
            err-=0.25
            dd, dl = detect(frame)
            fo.drawtext(f'mrkr det in {int(interval()*1000)}')
        else:
            dd = {}
            fo.drawtext(f'mrkr not in {int(interval()*1000)}')

        # if transform is not None:
        # to_interp = {}
        for k,v in list(ind.items()): # for each detection in cache
            if k not in dd: # if not detected in this frame
                if transform is not None:
                    v.corners = transform.transform(v.corners)
                    v.update()

                v.iage+=1
                if (v.iage>16)\
                    or (v.cxy[0]<0) or (v.cxy[0]>frame.shape[1])\
                    or (v.cxy[1]<0) or (v.cxy[1]>frame.shape[0]):
                    del ind[k]

        ind.update(dd)
        tags = ind

        fo.draw(lambda:mark_detections(frame, tags.values()))
        return tags

    return aruco_tracker

def tabletop_square_matcher_gen():
    tabletop_square_transform = tst = PerspectiveTransform()
    fail_counter = 0

    from shelftest import get_shelf
    with get_shelf() as data:
        if 'tst' in data:
            tst = data['tst']

    aruco_tracker = aruco_tracker_gen()

    def tabletop_square_matcher(fo):
        nonlocal tst, aruco_tracker, fail_counter
        frame = fo.frame

        tags = aruco_tracker(fo)

        # draw lines between tag 0-4
        # tagidx 指的是，放在桌子上，连接成正方形的，四个标记的编号。通常为 0,1,2,3
        # 取其他值时仅作测试用。
        tagidx = [0, 1, 2, 3]
        # tagidx = [22,24,42,40]

        l2d = []
        for p0i, p1i in [(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)]:
            try:
                p0 = tags[tagidx[p0i]]
                p1 = tags[tagidx[p1i]]
            except:
                continue
            else:
                l2d.append([p0.cxy, p1.cxy])
        if l2d:
            l2d = (np.array(l2d, dtype='float32')*4).astype('int32')
            fo.draw(lambda: cv2.polylines(
                frame, l2d, isClosed=False,
                color = (0,0,0), thickness=2, shift=2,
                lineType = cv2.LINE_AA,
            ))
            fo.draw(lambda: cv2.polylines(
                frame, l2d, isClosed=False,
                color = (255,255,200), thickness=1, shift=2,
                lineType = cv2.LINE_AA,
            ))

        if 0:
            # attempt to solve for cross point given 4 points
            p5xy = None
            if sum((1 if i in tags else 0 for i in tagidx))==4:
                x1,y1,x3,y3,x2,y2,x4,y4 = [
                    tags[tagidx[a]].cxy[b] for a in [0,1,2,3] for b in [0,1]]

                d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                if d!=0:
                    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/d
                    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/d

                    p5xy = np.array([px,py])


        # # draw from center to the right of tag 5
        # if 5 in tags:
        #     t5 = tags[5]
        #     t5r = t5.cxy + t5.rxy * 3.2
        #     fo.draw(lambda:
        #         dwsline(frame,
        #             (int(t5r[0]*4), int(t5r[1]*4)),
        #             (int(t5.cxy[0]*4), int(t5.cxy[1]*4)),
        #             color = (255,255,200), shift=2, thickness=2,
        #         )
        #     )

        # find transform from tag 0-4 to unit square
        found = [] # sources
        unit_square = [] # targets

        for idx, i in enumerate(tagidx):
            if i in tags:
                found.append(tags[i].cxy)
                unit_square.append(unit_square_coords[idx])

        # if p5xy is not None:
        #     # print(p5xy)
        #
        #     fo.draw(lambda:dwscircle(frame,
        #         (int(p5xy[0]), int(p5xy[1])),
        #         6,
        #         color=(255,50,128),
        #         shift=0, shadow=False, thickness=1))
        #
        #     found.append(p5xy)
        #     unit_square.append(np.array([.5, .5]))

        at = PerspectiveTransform()
        # at = AffineTransform()
        if len(found)>=4:
            # force least squares
            at.estimate_from(np.array(found), np.array(unit_square),
            # at.estimate_from(np.array(unit_square), np.array(found),
            method=0)

            if at.has_solution():
                tst = at

                if fail_counter == 40:
                    with get_shelf() as data:
                        data['tst'] = tst
                        print('tst saved')

            fail_counter = 0

        else: # no solution now
            fail_counter=min(40,fail_counter+1)


        # draw the transform found by converting a cross into screen space
        if tst.has_solution():
            fo.drawtext(f'tst got solution')

            # ns = nsplits = 5
            # for i in range(ns):
            #     for j in range(ns):
            #         new_p = tst.transform(np.array([i,j])/(ns-1), inverse=True)
            #
            #         fo.draw(lambda p1=new_p:
            #             dwscircle(frame,
            #                 (int(p1[0]), int(p1[1])),
            #                 5,
            #                 color=(255,50,128),
            #                 shift=0, shadow=False, thickness=2)
            #         )

            cl = []
            for i in range(5):
                k = i/4
                cl.append([[k, 0], [k, 1]])
                cl.append([[1, k], [0, k]])

            cross_lines = np.array(cl)
            # cross_lines = np.array([[[.5,0.], [.5,1.]], [[0.,.5], [1.,.5]]])
            cross_lines = tst.transform(cross_lines, inverse=True)*4
            cross_lines = cross_lines.astype('int32')

            fo.draw(lambda: cv2.polylines(
                frame, cross_lines, isClosed=False,
                color = (0,0,0), thickness=2, shift=2,
                lineType = cv2.LINE_AA,
            ))
            fo.draw(lambda: cv2.polylines(
                frame, cross_lines, isClosed=False,
                color = (255,200,200), thickness=1, shift=2,
                lineType = cv2.LINE_AA,
            ))


            # compensate perspective error
            for k in [] and tags:
                tag = tags[k]

                # back transform
                trans_corners = tst.transform(tag.corners, inverse=True)
                newtag = Detection('99',trans_corners)
                tag.rexy = tst.transform(newtag.rexy)
                tag.uexy = tst.transform(newtag.uexy)

        else:
            fo.drawtext('tst no solution')

        return tags, tst

    return tabletop_square_matcher

if __name__ == '__main__':
    # cl = camloop(aruco_tracker_gen(), threaded=True)
    cl = camloop(tabletop_square_matcher_gen(), threaded=True)

    while 1:
        cl.update()
        # print(cl.result)
        # time.sleep(0.1)

    # from tkfun2 import *
    # tk_imshow = tk_imshow_gen('camfeed')

    # def refresher():
    #     while 1:
    #         cl.update_f(lambda frame: tk_imshow(frame))
    #         time.sleep(0.1)

    # run_threaded(refresher)
    #
    # mainloop()


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
