"""
Geometric utilities for CAD2Sees.

Functions for polygon calculations, point-in-polygon tests, and
spatial analysis.
"""

import math
import numpy as np


def calculate_polygon_area(vertices):
    """
    Calculate polygon area using the shoelace formula.
    
    Args:
        vertices (list): List of (x, y) coordinate tuples
        
    Returns:
        float: Polygon area (always positive)
    """
    n = len(vertices)
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += (x1 * y2 - y1 * x2)
    area = abs(area) / 2.0
    return area


def polar_angle(point, points):
    """
    Calculate polar angle of a point relative to centroid of points.
    
    Args:
        point (tuple): Target point (x, y) coordinates
        points (list): List of (x, y) coordinate tuples for centroid
        
    Returns:
        float: Polar angle in radians (-π to π)
    """
    x_total = sum(x for x, _ in points)
    y_total = sum(y for _, y in points)
    centre = (x_total / len(points), y_total / len(points))
    x, y = point
    angle = math.atan2(y - centre[1], x - centre[0])
    return angle


def nearestCol(givenX, givenY, ColData):
    """
    Find the nearest column point to given coordinates.
    
    Args:
        givenX (float): Target x-coordinate
        givenY (float): Target y-coordinate
        ColData (pandas.DataFrame): DataFrame with 'X', 'Y', 'PointID' columns
        
    Returns:
        int/str: PointID of the nearest column point
    """
    min_point_id = None
    min_distance = float('inf')
    
    for _, row in ColData.iterrows():
        cur_x = row['X']
        cur_y = row['Y']
        distance = ((givenX - cur_x)**2 + (givenY - cur_y)**2)**0.5
        
        if distance < min_distance:
            min_point_id = row['PointID']
            min_distance = distance
            
    return min_point_id


def isin_polygon_vector(x, y, polygon):
    """
    Check if points are inside a polygon using vectorized operations.
    
    Args:
        x (array-like): Array of x-coordinates to test
        y (array-like): Array of y-coordinates to test
        polygon (list): List of (x, y) coordinate tuples for polygon vertices
        
    Returns:
        numpy.ndarray: Boolean array indicating if each point is inside
    """
    x = np.asarray(x)
    y = np.asarray(y)
    result = np.zeros_like(x, dtype=bool)

    n = len(polygon)
    p1x, p1y = polygon[0]
    
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]

        # Skip horizontal edges
        dy = p2y - p1y
        if dy == 0:
            p1x, p1y = p2x, p2y
            continue

        # Calculate x intersection for non-horizontal edges
        with np.errstate(divide='ignore', invalid='ignore'):
            xinters = (y - p1y) * (p2x - p1x) / dy + p1x

        # Only consider valid intersections
        valid_y = (y > min(p1y, p2y)) & (y <= max(p1y, p2y))
        valid_x = x <= max(p1x, p2x)
        valid_inters = x <= xinters

        # Combine all conditions with proper NaN handling
        inside = valid_y & valid_x & valid_inters
        inside = np.nan_to_num(inside, nan=False)

        result ^= inside
        p1x, p1y = p2x, p2y

    return result


def isin_polygon(x, y, polygons):
    """
    Check if a point is inside any of the given polygons.
    
    Args:
        x (float): x-coordinate of the point to test
        y (float): y-coordinate of the point to test
        polygons (list): List of polygons (each polygon is a list of
                        (x, y) tuples)
        
    Returns:
        int: Number of polygons that contain the point
    """
    count = 0
    
    for polygon in polygons:
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            
            if (y > min(p1y, p2y) and y <= max(p1y, p2y) and
                    x <= max(p1x, p2x) and p1y != p2y):
                
                xinters = ((y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x)
                
                if p1x == p2x or x <= xinters:
                    inside = not inside
                    
            p1x, p1y = p2x, p2y
            
        if inside:
            count += 1
            
    return count
