from arucofun import *

# some love from https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

marker_type = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

def marker_gen():
    tags = []
    tagsize = ts = 96
    outertagsize = ots = 96+48

    for i in range(50):
        id = i

        tag = np.zeros((ts, ts, 1), dtype="uint8")
        outertag = np.zeros((ots, ots, 1), dtype="uint8") + 255

        cv2.aruco.drawMarker(marker_type, id, tagsize, tag, 1)


        # for j in range(3):
        #     tag = cv2.blur(tag, (5,5))
        #     tag = cv2.blur(tag, (5,5))
        #     fac = 2
        #
        #     tag = tag * 0.95
        #
        #     tag = np.clip(tag.astype('float32') * fac - 256*(fac-1)*0.5, 0, 255
        #         ).astype('uint8')
            # res_thre, tag = cv2.threshold(tag, 127, 255, cv2.THRESH_BINARY)

        half = (ots - ts) //2
        outertag[half:ts+half, half:ts+half, :] = tag

        cv2.putText(outertag, f'{id:2d}', (32, 16), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0,), 1, cv2.LINE_AA)

        tags.append(outertag)

    tags = np.array(tags)

    # write the generated ArUCo tag to disk and then display it to our
    # screen

    # cv2.imwrite(args["output"], tag)


    # def batch_image_to_array(arr, margin=1, color=None, aspect_ratio=1.1, width=None, height=None):

    panel = vis.batch_image_to_array(tags, margin=16, color=240, aspect_ratio=1.1, width = None)

    vis.show_autoscaled(panel, name = 'panel', limit = 1600)

    cv2.imwrite('arucos.png', panel)
    # cv2.imshow("ArUCo Tag", tag)
    cv2.waitKey(0)

# def line(image, p1, p2, color):
#     cv2.line(image, p1,p2,(0,0,0),3,cv2.LINE_AA)
#     cv2.line(image, p1,p2,color,2,cv2.LINE_AA)

marker_gen()

def image_demo():
    im = cv2.imread('aruco_irl.jpg')

    detection_d, detection_l = detect(im)
    mark_detections(im, detection_l)

    vis.show_autoscaled(im, limit=1300)
    cv2.waitKey(0)

image_demo()
