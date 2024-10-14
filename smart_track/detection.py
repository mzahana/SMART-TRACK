#!/usr/bin/env python3
import numpy as np
import cv2
import math
import time

class DroneDetector:
    def __init__(self,area_bounds: list[int],
                 circular_bounds: list[float],
                 convexity_bounds: list[float],
                 d_group_max: int,
                 min_group_size: int,
                 max_cam_depth: float,
                 depth_scale_factor: float,
                 depth_step: float,
                 debug: bool):

        self.camera_info_ = None
        """
        Contour constraints
        """
        # params = {"area_bounds": [390, 10000], "circ_bounds": [0.3, 0.99], "conv_bounds": [0.7, 1.0], "d_group_max": 50, "min_group_size": 4, "max_cam_depth": 20.0, "depth_scale_factor": 1.0, "depth_step": 2}

        self.area_bounds_ =  area_bounds #[100, 1e4] # in pixels
        self.circ_bounds_ = circular_bounds # from 0 to 1
        self.conv_bounds_ = convexity_bounds # from 0 to 1
        self.d_group_max_ = d_group_max # maximal contour grouping distance in pixels
        self.min_group_size_ = min_group_size # minimal number of contours for a group to be valid
        self.max_cam_depth_ = max_cam_depth # Maximum acceptable camera depth values
        self.depth_scale_factor_ = depth_scale_factor # Scaling factor to make depth values in meters
        self.depth_step_ = depth_step
        self.debug_ = debug

    def depthTo3D(self, detections, depths):
        """
        @brief Computes 3D projections of detections in the camera frame (+X-right, +y-down, +Z-outward)
        @param detections : xy coordinates in 2D camera frame
        @param depths : Depths of detections in meters in camerra frame
        @return positions : 3D projections in camera frame
        """
        positions = []
        if self.camera_info_ is None:
            print("Camera intrinsic parameters are not available. Skipping 3D projections.")
            return positions

        fx = self.camera_info_['fx']
        fy = self.camera_info_['fy']
        cx = self.camera_info_['cx']
        cy = self.camera_info_['cy']
        for i in range(len(detections)):
            u = detections[i][1] # horizontal image coordinate
            v = detections[i][0] # vertical image coordinate
            d = depths[i] # depth

            x = d*(u-cx)/fx
            y = d*(v-cy)/fy

            p = [x,y,d]
            positions.append(p)

        return positions

    def preProcessing(self, img):
        """
        Pre-process input depth image.
        Finds list of contours (and their features) of a set of thresholded binary images.

        @param img: depth image
        @return valid_contours_list: List of valid contours in different thresholded images.
        @return contours_depths_list: List of depth values of each contour, for different thresholded images.
        @return contours_centers_list: List of each contour center, for different thresholded images.
        """
        t1 = time.time()
        if self.debug_:
            print("[preProcessing] Type of img:", type(img))
        img[np.isnan(img)] = self.max_cam_depth_ # Remove NaN values with the maximum distance provided by the camera
        img[np.isinf(img)] = self.max_cam_depth_
        max_depth_meter = img.max() * self.depth_scale_factor_
        min_depth_meter = img.min() * self.depth_scale_factor_

        if self.debug_:
            print( '[preProcessing] Max depth= {} Min depth = {}'.format( max_depth_meter, min_depth_meter) )

        # Normalize depth values
        norm_img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

        # Erosion
        eroded_img = self.erode(norm_img)

        thr_img_list = [] # List of all thresholded images
        imgs_cnt_list = [] # List of contours in each image
        valid_contours_list = []
        contours_depths_list = [] # Each element is a list of depths of contours found in the corresponding image
        contours_centers_list = []
        contours_radii_list = []
        imgs_cnt_features = [] # List of controus' features for each image

        # Loop through list of depth thresholds [ meters]
        min_threshold = int(min_depth_meter)+1
        max_threshold = int(max_depth_meter)
        # depth_range = range(min_threshold, max_threshold, self.depth_step_)
        depth_range = np.linspace(min_depth_meter+1.0, max_depth_meter, math.floor((max_depth_meter-min_depth_meter+1/self.depth_step_)))
        for depth in depth_range:
            # convert depth in meters to normalized value for OpenCV processing
            normalized_d = self.linearMap(depth, [min_depth_meter, max_depth_meter], [0., 1.])

            # Apply thresholding to the eroded image
            thr_img = self.thresholding(eroded_img, normalized_d)
            not_eroded_thr_img = self.thresholding(norm_img, normalized_d)
            thr_img_list.append(thr_img)

            # Extract contours and their features from the thresholded image
            #contours, cnt_features = self.getContours(thr_img)


            #imgs_cnt_list.append(contours)
            #imgs_cnt_features.append(cnt_features)

            # Find valid contours
            valid_contours, contours_depths, contours_centers, contours_radii = self.getValidContours2(thr_img, img, not_eroded_thr_img)
            if len(valid_contours) > 0:
                valid_contours_list.append(valid_contours)
                contours_depths_list.append(contours_depths)
                contours_centers_list.append(contours_centers)
                contours_radii_list.append(contours_radii)
                
            # else:
            #     print("No valid contours found at depth {}".format(depth))
        # print("Number of valid contours lists: {}".format(len(valid_contours_list)))

        # Extract valid detections
        valid_detections = []
        valid_depths = []
        valid_radii = []
        if len(contours_centers_list) > 0 :
            valid_detections, valid_depths, valid_radii = self.getValidDetections(contours_centers_list, contours_depths_list, contours_radii_list)
            if self.debug_:
                print('[preProcessing] Number of valid detections  = ', len(valid_detections))
                print('[preProcessing] Centers of valid detections  = ', valid_detections)
                print('[preProcessing] Depths of valid detections  = ', valid_depths)
        else:
            if self.debug_:
                print('[preProcessing] No contours found!')

        dt = time.time() - t1
        if self.debug_:
            print('[preProcessing] Detection extraction time = ', dt)

            #cv2.imshow("Thresholded image window: depth = " + str(depth), thr_img)

        
        # Draw image with detections
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,450)
        fontScale              = 0.6
        fontColor              = (0,0,255) # Red
        fontThickness          = 1
        lineType               = cv2.LINE_AA

        # cv2.imshow("Depth image window", norm_img)

        # cv2.imshow("Eroded image window", eroded_img)
        # middle_idx = int(len(thr_img_list)/2)
        # cv2.imshow("Thresholded image window", thr_img_list[middle_idx])

        # Draw valid detections
        backtorgb = img
        if len(valid_detections) > 0:
            # backtorgb = cv2.cvtColor(norm_img,cv2.COLOR_GRAY2RGB)
            for i in range(len(valid_detections)):
                center = valid_detections[i]
            # for center in valid_detections:
                backtorgb = self.drawDetectionMarker(backtorgb, center, valid_radii[i])
                # backtorgb = cv2.circle(backtorgb,(center[1],center[0]), valid_radii[i], (0,0,255), 2)
            backtorgb = cv2.putText(backtorgb,'Min Depth: {}, Max depth: {}'.format(min_depth_meter, max_depth_meter), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            fontThickness,
            lineType)
            # if self.show_debug_images_:
            #     cv2.imshow("Valid detections window: ", backtorgb)
            #     cv2.waitKey(1)

        return valid_detections, valid_depths, backtorgb

    def getValidDetections(self, contours_centers, contours_depths_list, contours_radii_list):
        """
        @brief Creates groups of contours that have close centers within predefined distance, and computes average center for each valid group

        @param contours_centers : list of countours center in each thresholded image
        @param contours_depths_list : Corresponding contours depths
        @param contours_radii_list List of controus radii, at each depth

        @return detections : List of centers of valid detections
        @return detections_depths : List of detections depths
        """

        # TODO : implement
        groups = [] # List of all groups of contours
        group = [] # single group of contours
        group_depths = []
        detections =[]
        detections_depths = []
        detections_radii = []
        group_idx = 0

        for i1 in range(len(contours_centers)):
            contours1 = contours_centers[i1]
            for j1 in range(len(contours1)):
                cnt1 = contours1[j1]
                group = []
                group_depths = []
                group_radii = []
                if cnt1 is not None:
                    group.append(cnt1)
                    group_depths.append(contours_depths_list[i1][j1])
                    contours_centers[i1][j1] = None

                    # compare the controur against all remaining ones
                    for i2 in range(len(contours_centers)):
                        contours2 = contours_centers[i2]
                        if i1 != i2: # skip comparing contours at the same threshold level
                            for j2 in range(len(contours2)):
                                cnt2 = contours2[j2]
                                if cnt2 is not None:
                                    p1 = np.array(cnt1)
                                    p2 = np.array(cnt2)
                                    dist = np.linalg.norm(p1-p2)
                                    if int(dist) <= self.d_group_max_: # compare distance between centers
                                        group.append(cnt2)
                                        group_depths.append(contours_depths_list[i2][j2])
                                        group_radii.append(contours_radii_list[i2][j2])
                                        contours_centers[i2][j2] = None

                if len(group) >= self.min_group_size_ :
                    np_g = np.array(group)
                    valid_center = sum(np_g) / len(group)
                    valid_depth = min(np.array(group_depths)) #sum(np.array(group_depths)) / len(group_depths)
                    valid_radius = sum(np.array(group_radii)) / len(group_radii)
                    detections.append(valid_center)
                    detections_depths.append(valid_depth)
                    detections_radii.append(valid_radius)

        # Extract valid groups
        # detections =[]
        # for i, g in enumerate(groups):
        #     if len(g) >= self.min_group_size_: # valid group of contours centers
        #         # find average center
        #         np_g = np.array(g)
        #         valid_center = sum(np_g) / len(g)
        #         detections.append(valid_center)

        return detections, detections_depths, detections_radii

    def getContours(self, img):
        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        # TODO: find valid contours in this method directly instead of using separate method getValidcontours() ????
        
        # List of features_dict of all contours; has the same length as contours list
        cnt_features_list = []

        for cnt in contours:
            features_dict = {'area': None, 'perimeter': None, 'circularity': None, 'convexity': None, 'depth': None}
            # Compute area
            features_dict['area'] = cv2.contourArea(cnt)
            # perimeter
            features_dict['perimeter'] = cv2.arcLength(cnt,True)
            # Circularity
            features_dict['circularity'] = 4.0*math.pi * features_dict['area'] / features_dict['perimeter']**2
            # Convexity (Solidity in OpenCV ?)
            cnt_area = features_dict['area']
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(cnt_area)/hull_area
            features_dict['convexity'] = solidity
            # Contour depth
            #features_dict['depth'] = self.getContourDepth()

            cnt_features_list.append(features_dict)


        return contours, cnt_features_list

    def getValidContours2(self, binary_img, orig_grayimg, not_eroded_thr_img):
        """
        @brief Finds valid contours in binary_img, their depths w.r.t orig_grayimg, and centers

        @param binary_img: Image after thresholding
        @param orig_grayimg: Gray scale image with depths in meters
        @param not_eroded_thr_img: Thresholded image without erosion. Used to get tight mask for better depth estimation for each contour.

        @return valid_contours: Valid countours
        @return valid_contours_depths: Depths of valid contours w.r.t orig_grayimg
        @return valid_contours_enters: Centers of valid contours in pixel coordinates
        @return valid_contours_radius Radii of valid contours
        """
        # contours, _ = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours, _ = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        valid_contours = []
        valid_contours_depths = []
        valid_contours_centers = []
        valid_contours_radius = []

        for cnt in contours:
            # Minimum enclosing circle
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            radius = int(radius)
            # Compute area
            area = cv2.contourArea(cnt)
            # perimeter
            perimeter = cv2.arcLength(cnt,True)
            # Circularity
            circularity = 4.0*math.pi * area / perimeter**2
            # Convexity (Solidity in OpenCV ?)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            convexity = solidity
            # Contour depth
            depth, coordinates = self.getContourDepth(cnt, orig_grayimg, not_eroded_thr_img)
            # Center
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = [cy, cx]
            # center = [coordinates[1], coordinates[0]]
            #depth = orig_grayimg[cx, cy]

            isAreaValid = area >= self.area_bounds_[0] and area <= self.area_bounds_[1]
            isCircValid = circularity >= self.circ_bounds_[0] and circularity <= self.circ_bounds_[1]
            isConvValid = convexity >= self.conv_bounds_[0] and convexity <= self.conv_bounds_[1]

            valid = isAreaValid and isCircValid and isConvValid
            if valid:
                valid_contours.append(cnt)
                valid_contours_depths.append(depth)
                valid_contours_centers.append(center)
                valid_contours_radius.append(radius)
            else:
                if self.debug_:
                    print('[getValidContours2] Area, circulariy, convexity contraints are not met')
                    print(f'[getValidContours2] Area constraint is not satisfied: area={area} bounds={self.area_bounds_}' )
                    print(f'Circularity constraint is not satisfied: circularity={circularity} bounds={self.circ_bounds_}' )
                    print(f'Convexity constraint is not satisfied: convexity={convexity} bounds={self.conv_bounds_}')

        return valid_contours, valid_contours_depths, valid_contours_centers, valid_contours_radius

    def getValidContours(self,contours, features):
        """
        @brief
        Finds valid contours which satisfy validity bounds defined in the __init__ method

        @param contours: List of contours
        @param features: List of features of contours

        @return valid_contours: List of valid contours. Returns None if no valid contour is found
        """
        cnt_N = len(contours)

        valid_contours = []
        for i in range(cnt_N):
            f = features[i]
            isAreaValid = f['area'] >= self.area_bounds_[0] and f['area'] <= self.area_bounds_[1]
            isCircValid = f['circularity'] >= self.circ_bounds_[0] and f['circularity'] <= self.circ_bounds_[1]
            isConvValid = f['convexity'] >= self.conv_bounds_[0] and f['convexity'] <= self.conv_bounds_[1]
            if isAreaValid and isCircValid and isConvValid:
                valid_contours.append(contours[i])
            
        if len(valid_contours) < 1: # No valid contour
            return  None
        
        return valid_contours

    def getContourDepth(self, cnt, img, not_eroded_thr_img):
        """
        @brief Computes the average intensity of all pixels inside a contour

        @param img: Input image in gray scale
        @param cnt: Input contour

        @return cnt_depth: Contour depth in the same unit as the input image
        """

        # Get tighter contour
        # M = cv2.moments(cnt)
        # cx = int(M['m10']/M['m00'])
        # cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(cnt)

        # reduce size of bounding rectangle to focus more on the object pixels
        # w=int(w/4)
        # h=int(h/4)
        # x = cx - w/2
        # y = cy - h/2
        # mask[y:y+h,x:x+w] = 255

        mask = np.zeros(img.shape,np.uint8)
        mask[y:y+h,x:x+w] = not_eroded_thr_img[y:y+h,x:x+w]
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        mask = np.zeros(img.shape,np.uint8)
        mask = cv2.drawContours(mask,contours,0,255,-1)

        #cv2.imshow("Mask", mask)
        #pixelpoints = np.transpose(np.nonzero(mask))
        cnt_depth = cv2.mean(img,mask = mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img,mask = mask)
        # return cnt_depth[0]
        return min_val, min_loc

    def erode(self, img):
        kernel = np.ones((15,15),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = 1)
        return erosion

    def thresholding(self, img, thr):
        t = thr
        # Sanity check on the threshold value
        if t < 0:
            print(f'Image threshold value is  {t} < 0. Setting threshold to 0.')
            t = 0.0

        if t > 1:
            print(f'Image threshold value {t} > 1. Setting threshold to 1.')
            t = 1.0

        _, threshold = cv2.threshold(img, t, 255, cv2.THRESH_BINARY_INV)
        return threshold

    def linearMap(self, val, in_range, out_range):
        # slope
        if (in_range[1] - in_range[0]) == 0.:
            return out_range[0]

        m = (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
        # Bias
        b = out_range[0] - m*in_range[0]

        # mapped value
        out_val = m*val + b
        return out_val

    def drawDetectionMarker(self, in_img, c, r):
        """
        @brief draws a circle with a cross around the target centerd at c with radius r
        
        Params
        --
        @param in_img Inout image
        @param c target center in image coordinates
        @param r radius of the target's enclosing circle

        Returns
        --
        @return out_img Output image with marker drawn on target at center c
        """
        r = int(r)
        color = (0, 0, 255) # Red
        thickness = 2
        cx = int(c[1])
        cy = int(c[0])
        img = cv2.circle(in_img,(cx,cy), r, color, int(thickness))

        # Line pointing to the right of the enclosing circle
        start_point = (cx+int(r/2), cy)
        end_point = (cx+int(r/2)+r, cy)
        img = cv2.line(img, start_point, end_point, color, thickness)

        # Line pointing to the left of the enclosing circle
        start_point = (cx-int(r/2), cy)
        end_point = (cx-int(r/2)-r, cy)
        img = cv2.line(img, start_point, end_point, color, thickness)

        # Line pointing to the top of the enclosing circle
        start_point = (cx, cy-int(r/2))
        end_point = (cx, cy-int(r/2)-r)
        img = cv2.line(img, start_point, end_point, color, thickness)

        # Line pointing to the bottom of the enclosing circle
        start_point = (cx, cy+int(r/2))
        end_point = (cx, cy+int(r/2)+r)
        img = cv2.line(img, start_point, end_point, color, thickness)

        return img
