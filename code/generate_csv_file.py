#!/usr/bin/env python

import cv2
import csv
import imutils
import numpy as pb
import argparse

coord = pb.array([0, 0])


def center_coord(filepath):
    global coord
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't retrieve frame - stream may have ended. Exiting..")
            break
        frame = imutils.resize(frame, width=1000)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_red = pb.array([160, 50, 50])
        h_red = pb.array([180, 255, 255])
        mask = cv2.inRange(hsv, l_red, h_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Res", res)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        coord = pb.vstack((coord, center))
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    coord = pb.delete(coord, 0, axis=0)
    cap.release()
    cv2.destroyAllWindows()  # destroy all windows
    return coord


# Arguments for video file path and csv file path
parse = argparse.ArgumentParser()
parse.add_argument(
    '--FilePath', default='./Data_Files/video1.mp4',
    help='Video file path')
parse.add_argument(
    '--CSVFilePath', default='./Data_Files/data1.csv',
    help='File path to save csv file')
Args = parse.parse_args()
file_path = Args.FilePath
CSVFile_path = Args.CSVFilePath

coord = center_coord(file_path)
header = [['x', 'y']]
with open(CSVFile_path, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(header)
    writer.writerows(coord)
