from __future__ import division

import cv2
import numpy
import os
import re

class Pipeline(object):
    
    def __init__(self):
        self._lower_hue_0 = numpy.array([80, 90, 100])
        self._lower_hue_1 = numpy.array([120, 255, 255])
        self._upper_hue_0 = numpy.array([80, 90, 100])
        self._upper_hue_1 = numpy.array([120, 255, 255])
        
        self._kernel_median_blur = 27
        self._kernel_dilate_mask = (9, 9)
        
        self._x = -1
        self._y = -1
        self._dx = 0
        self._dy = 0
        self._vx = 0
        self._vy = 0
        self._histdx = []
        self._histdy = []
        self._points = []
        self._max_points = 400
        self._min_change = 10
        self._min_veloxy = 2.0
        self._marker_ctr = None
        self._marker_tip = None
        
        self._fps = 20
        
        self._render_marker = True
        self._render_trails = True
        
        self._opencv_version = int(cv2.__version__.split('.')[0])
        
        return
    
    def _marker_segmentation(self, frame):
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        mask_0 = cv2.inRange(frame_hsv, self._lower_hue_0, self._lower_hue_1)
        mask_1 = cv2.inRange(frame_hsv, self._upper_hue_0, self._upper_hue_1)
        
        mask = cv2.addWeighted(mask_0, 1.0, mask_1, 1.0, 0.0)
        
        mask = cv2.medianBlur(mask, self._kernel_median_blur)
        
        mask = cv2.dilate(mask, self._kernel_dilate_mask)
        
        return mask
    
    def _marker_tip_identification(self, mask):
        #If you get an -215 Assertion Error, try changing the index of the next line from 0 to 1
        image = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = image if len(image) == 2 else image[1:3]
        
        #contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        
        if contours and len(contours) > 0:
            contour_max = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
            contour_roi = contour_max.reshape(contour_max.shape[0], contour_max.shape[2])
            contour_roi = sorted(contour_roi, key=lambda x:x[1])
            
            marker_tip = (contour_roi[0][0], contour_roi[0][1])
        else:
            contour_max = None
            marker_tip = None
        
        return [contour_max, marker_tip]
    
    def _trajectory_approximation(self, marker_tip, frame):
        image = None
        if marker_tip is None:
            if self._points:
                self.save_data(self._points)
            self._x = -1
            self._y = -1
            self._dx = 0
            self._dy = 0
            self._vx = 0
            self._vy = 0
            self._histdx = []
            self._histdy = []
            self._points = []
        else:
            if len(self._histdx) >= self._fps:
                self._histdx.pop(0)
            if len(self._histdy) >= self._fps:
                self._histdy.pop(0)
            if len(self._points) > self._max_points:
                self._points.pop(0)
            
            if self._x < 0 or self._y < 0:
                self._x, self._y = marker_tip
            self._dx = abs(marker_tip[0] - self._x)
            self._dy = abs(marker_tip[1] - self._y)
            self._histdx.append(self._dx)
            self._histdy.append(self._dy)
            if self._dx > self._min_change or self._dy > self._min_change:
                self._points.append(marker_tip)
            self._x, self._y = marker_tip
            
            self._vx = numpy.floor(sum(self._histdx[-self._fps:]) / self._fps)
            self._vy = numpy.floor(sum(self._histdy[-self._fps:]) / self._fps)
            
            nodes = len(self._points)
            if nodes > 1:
                image = numpy.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
                for i in range(nodes-1):
                    cv2.line(image, self._points[i], self._points[i+1], (255, 255, 255), 4, cv2.LINE_AA)
        
        return image, self._points
    
    def _render(self, frame):
        if not self._marker_ctr is None:
            cv2.drawContours(frame, self._marker_ctr, -1, (0, 255, 0), 1)
        if not self._marker_tip is None:
            cv2.circle(frame, self._marker_tip, 4, (255, 255, 0), -1)
        n = len(self._points)
        if n > 1:
            for i in range(n-1):
                cv2.line(frame, self._points[i], self._points[i+1], (255, 255, 0), 4, cv2.LINE_AA)
        
        return frame
    
    def run_inference(self, frame, engine='EN', mapping=True):
        mask = self._marker_segmentation(frame)

        self._marker_ctr, self._marker_tip = self._marker_tip_identification(mask)

        image, pts = self._trajectory_approximation(self._marker_tip, frame)

        if not image is None and self._vx < self._min_veloxy and self._vy < self._min_veloxy:
            
            self._x = -1
            self._y = -1
            self._dx = 0
            self._dy = 0
            self._vx = 0
            self._vy = 0
            self._histdx = []
            self._histdy = []
            self._points = []
            self._marker_ctr = None
            self._marker_tip = None
            
            self.save_data(pts)

        frame = self._render(frame)
        
        return frame
    
    def save_data(self, points):
        if len(points) > 10:
            if not os.path.exists('generated_data/'):
                    os.mkdir('generated_data/')
                
            c = 0
            files = []
            for filename in os.listdir('generated_data/'):
                if filename.endswith('.npy'):
                    files.append(filename)
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            files = sorted(files, key = alphanum_key)
            if files:
                c = files[len(files) - 1]
                c = c[:c.find('_')]
                c = int(c)
                c = c + 1
                print(files)
            
            while os.path.isfile('generated_data/' + str(c) + '_' + str(len(points)) + '.npy'):
                c = c + 1
            numpy.save('generated_data/' + str(c) + '_' + str(len(points)) + '.npy', points)
            print('Data saved as: ', 'generated_data/' + str(c) + '_' + str(len(points)) + '.npy')
