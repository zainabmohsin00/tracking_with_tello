import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from djitellopy import tello
from PIL import Image

######################
import torch
import torchvision.transforms as transforms
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from apn import APN_Model

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    # ...
    model = APN_Model()  # define the model architecture
    model.load_state_dict(torch.load('best_model.pt'))  # load the weights from the file
    model.eval()

    def preprocess_input(img):
        transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
        ])

        img = img.convert('RGB')
        img = transform(img).unsqueeze(0)
        return img

    def preprocess_images(image_paths):
        processed_images = []
        for path in image_paths:
            img = load_img(path, target_size=(224, 224))
            x = preprocess_input(img)
            processed_images.append(x)
        return processed_images

    def find_best_match(input_path, gallery_paths):
        input_image = preprocess_input(load_img(input_path, target_size=(224, 224)))
        gallery_images = preprocess_images(gallery_paths)
        input_embedding = model(input_image)

        best_similarity = -1
        best_match_path = None

        for i in range(len(gallery_images)):
            gallery_embedding = model(gallery_images[i])
            similarity = cosine_similarity(input_embedding, gallery_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_path = gallery_paths[i]

        return best_match_path

    def cosine_similarity(x, y):
        x_norm = x.norm(dim=1, keepdim=True)
        y_norm = y.norm(dim=1, keepdim=True)
        cosine_sim = torch.mm(x, y.t()) / torch.mm(x_norm, y_norm.t())
        return cosine_sim.detach().numpy()

    input_path = r'C:\Users\Ryzen\Documents\PROJECTS\yolov3DeepSort\yolov3_deepsort\images\zf.jpg'
    ground_img = cv2.imread(input_path)
    cv2.imwrite('detections/ground_img_' + '.jpg', ground_img)
    gallery_paths = []

    # Tello
    me = tello.Tello()

    me.connect()

    me.set_speed(100)
    me.takeoff()
    time.sleep(5)
    me.move_up(180)

    me.streamon()

    # me.send_control_command("keepalive")

    reidentified = False
    best_match_path = ''

    while not reidentified:
        frame = me.get_frame_read().frame

        while frame is None:
            frame = me.get_frame_read().frame
            continue

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Frame received")

        # frame = me.get_frame_read().frame

        height, width, channels = frame.shape

        # Constants for tracking
        max_x_offset = width // 7
        max_y_offset = height // 10

        mid_x = width // 2
        mid_y = height // 2

        # Definition of the parameters
        max_cosine_distance = 0.5
        nn_budget = None
        nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)

        yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')

        class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')

        try:
            vid = cv2.VideoCapture(int(FLAGS.video))
        except:
            vid = cv2.VideoCapture(FLAGS.video)

        out = None

        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
            list_file = open('detection.txt', 'w')
            frame_index = -1

        fps = 0.0
        count = 0
        frame_id = 0

        # _, img = vid.read()
        frame_count = 0
        frame_count += 1
        detection_count = 0

        ######################################################
        if frame is not None:
            img_copy = frame.copy()
            print("printing img copy")
            cv2.imwrite('detections/original_img_' + '.jpg', img_copy)

        if frame is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count += 1

        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []

        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(frame, boxes[0])
        features = encoder(frame, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        for detection in detections:
            x, y, w, h = detection.tlwh
            cropped_image = img_copy[int(y):int(y)+int(h), int(x):int(x)+int(w)]
            # Save the cropped image
            cv2.imwrite('detections/cropped_img' + str(detection_count) + '.jpg', cropped_image)
            gallery_paths.append('detections/cropped_img' + str(detection_count) + '.jpg')
            detection_count += 1

        best_match_path = find_best_match(input_path, gallery_paths)
        if best_match_path != '':
            reidentified = True
        d_count = int(best_match_path.split('_img')[-1].split('.')[0])
        detection_to_track = [detections[d_count]]

    # @@@@@@@@@@@@@@@@@@@ Tracking Initialised   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Call the tracker
    tracker.predict()
    tracker.update(detection_to_track)

    target_track_id = tracker.tracks[0].track_id

    for track in tracker.tracks:
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                      (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                    (255, 255, 255), 2)

    cv2.imwrite('detections/c_' + str('detected') + '.jpg', frame)
    cv2.imwrite('detections/c_' + '.jpg', img_copy)
    cv2.imshow('output', frame)

    if FLAGS.output:
        out.write(frame)
        frame_index = frame_index + 1
        list_file.write(str(frame_index) + ' ')
        if len(converted_boxes) != 0:
            for i in range(0, len(converted_boxes)):
                list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(
                    converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
        list_file.write('\n')

    while True:
        time.sleep(0.1)
        me.rotate_clockwise(1)

        img = me.get_frame_read().frame
        frame_id += 1

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if track.track_id == target_track_id:
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
                cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)
                cv2.imshow('output', img)

                center_x = (int(bbox[0]) + int(bbox[2]))/2
                center_y = (int(bbox[1]) + int(bbox[3]))/2
                # Adjusting Tello
                # Left
                if center_x - mid_x < -max_x_offset:
                    # me.move_left(50)
                    me.rotate_counter_clockwise(10)
                # Right
                elif center_x - mid_x > max_x_offset:
                    # me.move_right(50)
                    me.rotate_clockwise(10)
                # Forward
                if center_y - mid_y < -max_y_offset:
                    me.move_forward(100)
                # Backward
                elif center_y - mid_y > max_y_offset:
                    me.move_back(100)

        # print fps on screen
        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)

        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(converted_boxes) != 0:
                for i in range(0, len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(
                        converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            me.land()
            break

    vid.release()
    if FLAGS.output:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass