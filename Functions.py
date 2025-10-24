# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    # Convert points to numpy arrays if they are MediaPipe landmarks
    a = np.array([a.x, a.y]) if hasattr(a, 'x') else np.array(a)
    b = np.array([b.x, b.y]) if hasattr(b, 'x') else np.array(b)
    c = np.array([c.x, c.y]) if hasattr(c, 'x') else np.array(c)
    
    # Calculate vectors
    ab = a - b
    cb = c - b
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(ab, cb)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_cb = np.linalg.norm(cb)
    
    # Calculate angle in radians and convert to degrees
    angle_radians = np.arccos(dot_product / (magnitude_ab * magnitude_cb))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Function to calculate tilt (angle of a line relative to the horizontal axis)
def calculate_tilt(a, b):
    # Convert points to numpy arrays if they are MediaPipe landmarks
    a = np.array([a.x, a.y]) if hasattr(a, 'x') else np.array(a)
    b = np.array([b.x, b.y]) if hasattr(b, 'x') else np.array(b)
    
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    tilt_radians = math.atan2(dy, dx)
    tilt_degrees = math.degrees(tilt_radians)
    return tilt_degrees

# Function to calculate speed (pixels per frame)
def calculate_speed(prev_point, curr_point, frame_rate):
    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]
    distance = math.sqrt(dx**2 + dy**2)
    speed = distance * frame_rate
    return speed

# Function to calculate arc curvature
def calculate_arc_curvature(points):
    if len(points) < 3:
        return 0
    
    # Simple curvature estimation (change in angle)
    angles = []
    for i in range(1, len(points)-1):
        v1 = (points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
        v2 = (points[i+1][0] - points[i][0], points[i+1][1] - points[i][1])
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 > 0 and mag2 > 0:
            angle = math.acos(max(min(dot / (mag1 * mag2), 1.0), -1.0))
            angles.append(math.degrees(angle))
    
    return np.mean(angles) if angles else 0

# Function to calculate wrist angular rotation
def calculate_wrist_rotation(prev_wrist_pos, curr_wrist_pos, prev_elbow_pos, curr_elbow_pos):
    """
    Calculate wrist angular rotation in degrees per second
    Based on the change in angle between elbow-wrist vector and horizontal axis
    """
    if prev_wrist_pos is None or prev_elbow_pos is None:
        return 0
    
    # Calculate previous vector from elbow to wrist
    prev_vector = (prev_wrist_pos[0] - prev_elbow_pos[0], 
                   prev_wrist_pos[1] - prev_elbow_pos[1])
    
    # Calculate current vector from elbow to wrist
    curr_vector = (curr_wrist_pos[0] - curr_elbow_pos[0],
                   curr_wrist_pos[1] - curr_elbow_pos[1])
    
    # Calculate angles of both vectors relative to horizontal
    prev_angle = math.degrees(math.atan2(prev_vector[1], prev_vector[0]))
    curr_angle = math.degrees(math.atan2(curr_vector[1], curr_vector[0]))
    
    # Calculate angular difference (considering circular nature)
    angle_diff = curr_angle - prev_angle
    
    # Normalize to [-180, 180] range
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    
    return angle_diff

# Function to calculate angular velocity
def calculate_angular_velocity(angle_differences, fps):
    """
    Calculate angular velocity from angle differences
    """
    if not angle_differences:
        return 0
    
    # Average angle difference per frame
    avg_angle_diff = np.mean(angle_differences)
    
    # Convert to degrees per second
    angular_velocity = avg_angle_diff * fps
    
    return angular_velocity

# Function to detect ball using simple color-based detection
def detect_ball_simple(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color (tennis ball)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Only consider reasonably sized objects
        if w > 10 and h > 10:
            return [(x + w/2, y + h/2, 0.7)]
    
    return None

# Function to detect racquet using shape detection
def detect_racquet_simple(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    racquets = []
    for contour in contours:
        # Filter by area and aspect ratio
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Racquet-like aspect ratio
            if 0.3 < aspect_ratio < 3.0:
                racquets.append((x + w/2, y + h/2, 0.6))
    
    return racquets