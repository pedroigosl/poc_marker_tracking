import math
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

COLOR_AREA = 4*4
WHITE_AREA = 10*10

# HSV limits
COLOR_LIMITS = {
    'red':{
        'min_color': (160,50,80),
        'max_color': (180, 255, 255),
        'min_area': COLOR_AREA,
    },
    'green':{
        'min_color': (30, 50, 80),
        'max_color': (80, 255, 255),
        'min_area': COLOR_AREA,
    },
    'blue':{
        'min_color': (80, 70, 120),
        'max_color': (130, 255, 255),
        'min_area': COLOR_AREA,
    },
    'white':{
        'min_color': (0,0,150),
        'max_color': (180, 50, 255),
        'min_area': WHITE_AREA,
    }
}

LEGEND_PARAMS = {
    'fontScale': 1,
    'fontFace': cv2.LINE_AA,
    'thickness': 2,
    'org': (50, 50),
}

DEFAULT_TEXT_COLOR  = (0, 255, 0)

MARKER_ERROR = 0.5

def get_color_masks(img_map, color_limits = COLOR_LIMITS):

    for color, limits in color_limits.items():
        mask = img_map['hsv']['img']
        mask = cv2.inRange(mask, limits['min_color'], limits['max_color'])
        mask = cv2.medianBlur(mask, ksize=3)

        img_tag = f"mask_{color}"
        img_map[img_tag] = {
            'img': mask,
            'lgd': img_tag,
            'lgd_color': DEFAULT_TEXT_COLOR,
        }
        
    return img_map

def calculate_contours(img_map, color_limits = COLOR_LIMITS):
    contour_map = {}
    for color in ['red', 'green', 'blue', 'white']:
        img_tag = f"mask_{color}"
        mask = img_map[img_tag]['img']
        # Calculating external contours 
        contours, hierarchy = cv2.findContours(
            image=mask, 
            mode=cv2.RETR_CCOMP, 
            method=cv2.CHAIN_APPROX_SIMPLE)

        gray2rgb_img = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        
        contour_img = cv2.drawContours(
            image=gray2rgb_img, 
            contours=contours, 
            contourIdx=-1, 
            color=(0, 255, 0), 
            thickness=5, 
            lineType=cv2.LINE_AA
        )
        
        # Getting contour centroids
        centroids = []
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)

            # Checking if contour has valid area
            weight = int(M['m00'])
            if (weight >= color_limits[color]['min_area']):
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                centroids.append([cx, cy, weight, i])

        # Marking all centroids
        for centroid in centroids:
            cv2.circle(contour_img, (centroid[0], centroid[1]), 10, (255, 0, 0), -1)

        contour_map[color] = {
            'contours': contours,
            'hierarchy': hierarchy,
            'centroids': centroids,
        }
        img_map[img_tag]['img'] = contour_img
        
    return img_map, contour_map

def find_marker(img_map, contour_map, error_threshold = MARKER_ERROR):
    # Validating background contours
    best_fit = {
        'id': -1,
        'error': None,
        'area': None,
        'contours': None,
        'is_valid': False,
    }
    bkgd_contour_data = contour_map['white']
    for bkgd_centroid in bkgd_contour_data['centroids']:        
        contour_id = bkgd_centroid[-1]
        contour_area = bkgd_centroid[2]
        bkgd_contour = bkgd_contour_data['contours'][contour_id]
        
        # No need to check emcompassing contours
        best_area = best_fit['area']
        if ((best_fit['id'] != -1) and (contour_area > best_area*1.05)):
            continue
        
        # Square contour check
        is_square = False        
        epsilon = 0.03*cv2.arcLength(bkgd_contour,True)
        approx_contour = cv2.approxPolyDP(bkgd_contour,epsilon, True)
        
        if len(approx_contour) == 4:
            contour_bbox = cv2.minAreaRect(approx_contour)
            bbox_dims = contour_bbox[1]
            bbox_aspect_ratio = bbox_dims[0] / bbox_dims[1]
            if 0.8 < bbox_aspect_ratio < 1.2:
                is_square = True
            
        if not is_square:
            continue
        
        # Check for RGB contours inside white contour
        inside_contours = {}
        all_colors = True
        for color in ['red', 'green', 'blue']:
            inside_contours[color] = {}
            inside_contours[color]['centroids'] = []
            for color_centroid in contour_map[color]['centroids']:
                # Point's distance to contour ( > 0: Indide | < 0: Outside | = 0: Over contour)
                relative_distance = cv2.pointPolygonTest(
                    contour= bkgd_contour, 
                    pt= (color_centroid[0], color_centroid[1]), 
                    measureDist= False #True
                )
                
                if relative_distance > 0:
                    inside_contours[color]['centroids'].append(color_centroid)
            
            
            if len(inside_contours[color]['centroids']) > 0:
                # Getting main centroid from weighted average over contour areas
                inside_centroids = np.array(inside_contours[color]['centroids'])
                avg_centroid = [
                    int(sum(inside_centroids[:,2]*inside_centroids[:,0])/sum(inside_centroids[:,2])),
                    int(sum(inside_centroids[:,2]*inside_centroids[:,1])/sum(inside_centroids[:,2]))
                ]
                inside_contours[color]['center'] = avg_centroid
                
            # Missing a color in marker
            else:
                all_colors = False
                break
        if not all_colors:
            continue
        
        # Calculating Marker geometry errors using Pythagorean theorem
        center_R = inside_contours['red']['center'][:2]
        center_G = inside_contours['green']['center'][:2]
        center_B = inside_contours['blue']['center'][:2]
        
        red_green_dist = math.dist(center_R, center_G)
        red_blue_dist = math.dist(center_R, center_B)
        green_blue_dist = math.dist(center_G, center_B)
        
        # Error over Hypotenuse
        error_abs = abs(green_blue_dist - math.sqrt(math.pow(red_green_dist,2) + math.pow(red_blue_dist,2)))
        error_rel = error_abs/green_blue_dist
        
        # Geometry errors under error margin
        error = error_rel
        is_marker = error_rel <= error_threshold

        # First marker
        if best_fit['id'] == -1:
            best_fit['id'] = contour_id
            best_fit['error']  = error
            best_fit['area'] = contour_area
            best_fit['contours'] = inside_contours
        
        if ((error <= best_fit['error']) and (contour_area <= best_fit['area']) and is_marker):
            best_fit['id'] = contour_id
            best_fit['error']  = error
            best_fit['area'] = contour_area
            best_fit['contours'] = inside_contours
            best_fit['is_valid'] = True

    return img_map, bkgd_contour_data, best_fit

def draw_annotations(img_map, bkgd_contours, best_fit):
    marker_img = img_map['baseline']['img'].copy()
    # No marker found
    if best_fit['id'] == -1:
        
        img_lgd = f"Marker Missing!!"   
        img_map['marker'] = {
        'img': marker_img,
        'lgd': img_lgd,
        'lgd_color': (255, 0, 0),
        }
        return img_map
    
    # Drawing bounding box
    marker_img = cv2.drawContours(
        image= marker_img, 
        contours=bkgd_contours['contours'], 
        contourIdx=best_fit['id'], 
        color=(0, 255, 0), 
        thickness=5, 
        lineType=cv2.LINE_AA
    )
    
    # Drawing color centroids
    inside_contours = best_fit['contours']
    for color, values in inside_contours.items():
        for centroid in values['centroids']:
            cv2.circle(marker_img, (centroid[0], centroid[1]), 5, (255, 0, 0), -1)
        cv2.circle(marker_img, values['center'], 5, (0, 0, 255), -1)
    
    # Drawing RGB connecting lines
    center_R = inside_contours['red']['center'][:2]
    center_G = inside_contours['green']['center'][:2]
    center_B = inside_contours['blue']['center'][:2]
    
    cv2.line(marker_img, center_G, center_B, (255, 255, 0), 3)
    cv2.line(marker_img, center_G, center_R, (255, 255, 0), 3)
    cv2.line(marker_img, center_R, center_B, (255, 255, 0), 3)
    
    # Designing image legend
    img_lgd = f"Marker Found! [error: {best_fit['error']*100:.2f}]"  
    lgd_color = DEFAULT_TEXT_COLOR
    # Error above threshold
    if not best_fit['is_valid']:
        lgd_color = (255, 255, 0)
        
    img_map['marker'] = {
    'img': marker_img,
    'lgd': img_lgd,
    'lgd_color': lgd_color,
    }
    
    return img_map

def get_frame(img_map):
    for img_tag, img_data in img_map.items():
        img = img_data['img']
        lgd = img_data['lgd']
        color = img_data['lgd_color']
        
        img_map[img_tag]['img'] = cv2.cvtColor(cv2.putText(img, lgd, color= color, **LEGEND_PARAMS), cv2.COLOR_BGR2RGB)

    mask_img = [cv2.resize(img_map[img_tag]['img'], None, fx=1/4, fy=1/4) for img_tag in img_map.keys() if 'mask' in img_tag]
    mask_img = cv2.hconcat(mask_img)
    main_img = img_map['marker']['img']
    
    output_frame = cv2.vconcat([main_img, mask_img])
    
    return output_frame

def video_tracking(path, name, ext = '.mp4', codec= 'mp4v', render = False):
    cap = cv2.VideoCapture(path + name + ext)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_height = int(frame_height + frame_height*1/4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    full_path = path + name + '_tracking' + ext
    out = cv2.VideoWriter(full_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        img_map = {
            'baseline': {
                'img': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                'lgd': 'baseline',
                'lgd_color': DEFAULT_TEXT_COLOR,
            },
            'hsv': {
                'img': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                'lgd': 'hsv',
                'lgd_color': DEFAULT_TEXT_COLOR,
            },
        }
        
        img_map = get_color_masks(img_map)
        img_map, contour_map = calculate_contours(img_map)
        img_map, bkgd_contours, best_fit = find_marker(img_map, contour_map)
        img_map = draw_annotations(img_map, bkgd_contours, best_fit)

        out_frame = get_frame(img_map)
        
        out.write(out_frame)
        
        if render:
            cv2.imshow('Marker Tracking', out_frame)

            if cv2.waitKey(1) == ord('q'):
                break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = './data/'
    name = 'sample_video'
    render = False

    print(f"Processing: {path+name}")
    
    start_time = time.perf_counter()
    video_tracking(path, name, render = render)
    end_time = time.perf_counter()
    
    print(f"time elapsed: {end_time - start_time:.1f} seconds")
