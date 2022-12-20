"""odas.py
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

import pyttsx3 as tts
import datetime as date

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.boxes import BoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.video_writer import get_video_writer


WINDOW_NAME = 'ODAS'
    

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def audio_warning(start,elapsed,confs,boxes,cam,middle_check,engine):
    if start != 0:    
        elapsed = time.time() - start
        
        #print(start,elapsed)
        is_middle=False
        say_middle=False
        say_side=False
        print(len(confs))
        if len(confs) != 0:
            for i in range(len(confs)):
                print(i,len(confs))
                if ((boxes[i][2]>cam.img_width/3 and boxes[i][2]<2*cam.img_width/3) or (boxes[i][0]>cam.img_width/3 and boxes[i][0]<2*cam.img_width/3)):
                    is_middle = True
            print(is_middle)
            if is_middle == True and middle_check == True:
                start_mid_timer=time.time()
                pass
            elif is_middle == True and middle_check == False:
                middle_check=True
                if time.time()-start_mid_timer >= 5:
                    start_mid_timer=time.time()
                    start=time.time()
                    #print("midped")
                    engine.say('pedestrian on road')
                    engine.runAndWait()
            elif is_middle == False and elapsed >= 15 and time.time()-start_mid_timer >= 5:
                middle_check=False
                start=time.time()
                #print("sideped")
                engine.say('Pedestrian on sidewalk')
                engine.runAndWait()
            else:
                middle_check=False
        else:
            middle_check=False

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    start = 0
    start_mid_timer=0
    elapsed =60
    middle_check = False
    engine = tts.init()
    engine.setProperty("rate",150)
    color = (0, 0, 255)

    #record_out=get_video_writer(str(date.datetime.now()), cam.img_width, cam.img_height, fps=30)
    
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
            
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        audio_warning(start,elapsed,confs,boxes,cam,middle_check,engine)       

        img = vis.draw_boxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        #cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        cv2.rectangle(img, (int(cam.img_width/3), 0), (int(2*cam.img_width/3), cam.img_height), (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, img)
        #record_out.write(img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

def main():
    

    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')



    cls_dict = get_cls_dict(args.category_num)
    vis = BoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
    
    open_window(
        WINDOW_NAME, 'ODAS Camera',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, conf_th=0.5, vis=vis)


    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
