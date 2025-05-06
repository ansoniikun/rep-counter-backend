from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import io
from PIL import Image

tracking_started = False
ready_start_time = None
ready_hold_duration = 5  # seconds


tracking_enabled = True  # Controls whether to track reps
current_exercise = 'bicep_curl'  # Default exercise

app = Flask(__name__)
CORS(app)

# Mediapipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global rep counters
counter_left, stage_left = 0, None
counter_right, stage_right = 0, None

# Landmarks to ignore (face)
face_landmarks = {
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT,
}

# Helper: angle between three joints
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def is_ready_bicep_curl(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    return angle_left > 150 and angle_right > 150

def is_ready_squat(landmarks):
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    angle = calculate_angle(hip, knee, ankle)
    return angle > 160

def is_ready_bench_press(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Expect elbows to be bent when bar is at chest level
    return 70 <= angle_left <= 110 and 70 <= angle_right <= 110



def is_ready_deadlift(landmarks):
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    angle = calculate_angle(hip, knee, ankle)
    return angle > 160

def is_ready_shoulder_press(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Shoulder press ready = elbows at roughly 90° angles
    return 10 <= angle_left <= 90 and 10 <= angle_right <= 90



# Rep logic: Bicep Curls
def track_bicep_curls(landmarks):
    global counter_left, stage_left, counter_right, stage_right

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    if angle_left > 160:
        stage_left = "down"
    if angle_left < 30 and stage_left == "down":
        stage_left = "up"
        counter_left += 1

    if angle_right > 160:
        stage_right = "down"
    if angle_right < 30 and stage_right == "down":
        stage_right = "up"
        counter_right += 1

# Rep logic: Squats
def track_squats(landmarks):
    global counter_left, stage_left

    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    angle = calculate_angle(hip, knee, ankle)

    if angle > 160:
        stage_left = 'up'
    if angle < 90 and stage_left == 'up':
        stage_left = 'down'
        counter_left += 1

# Rep logic: Bench press
def track_bench_press(landmarks):
    global counter_left, stage_left, counter_right, stage_right

    # LEFT
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)

    if angle_left < 90:
        stage_left = "down"
    if angle_left > 160 and stage_left == "down":
        stage_left = "up"
        counter_left += 1

    # RIGHT
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    if angle_right < 90:
        stage_right = "down"
    if angle_right > 160 and stage_right == "down":
        stage_right = "up"
        counter_right += 1




# Rep logic: Deadlift
def track_deadlift(landmarks):
    global counter_left, stage_left

    # Use the left side of the body to track hip extension
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    angle = calculate_angle(hip, knee, ankle)

    if angle > 160:
        stage_left = "up"
    if angle < 90 and stage_left == "up":
        stage_left = "down"
        counter_left += 1

# Rep logic: shoulder press
def track_shoulder_press(landmarks):
    global counter_left, stage_left, counter_right, stage_right

    # LEFT SIDE
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)

    if angle_left > 130:
        stage_left = "up"
    if angle_left < 100 and stage_left == "up":
        stage_left = "down"
        counter_left += 1

    # RIGHT SIDE
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    if angle_right > 130:
        stage_right = "up"
    if angle_right < 100 and stage_right == "up":
        stage_right = "down"
        counter_right += 1





# Video generator
def gen_frames():
    global counter_left, counter_right, stage_left, stage_right

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                h, w, _ = image.shape
                for idx, landmark in enumerate(landmarks):
                    if mp_pose.PoseLandmark(idx) not in face_landmarks:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (cx, cy), 4, (255, 0, 0), -1)

                filtered_connections = [
                    (start, end) for (start, end) in mp_pose.POSE_CONNECTIONS
                    if start not in face_landmarks and end not in face_landmarks
                ]
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    filtered_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )

                if tracking_enabled:
                    global tracking_started, ready_start_time

                    # Select appropriate ready check
                    if current_exercise == 'bicep_curl':
                        is_ready = is_ready_bicep_curl(landmarks)
                    elif current_exercise == 'squat':
                        is_ready = is_ready_squat(landmarks)
                    elif current_exercise == 'bench_press':
                        is_ready = is_ready_bench_press(landmarks)
                    elif current_exercise == 'deadlift':
                        is_ready = is_ready_deadlift(landmarks)
                    elif current_exercise == 'shoulder_press':
                        is_ready = is_ready_shoulder_press(landmarks)
                    else:
                        is_ready = False

                    # Wait until user holds ready position
                    # Improved hold check with tolerance
                    if is_ready:
                        if ready_start_time is None:
                            ready_start_time = time.time()
                        elapsed = time.time() - ready_start_time

                        if elapsed >= ready_hold_duration:
                            # Show "Ready!" message
                            cv2.putText(image, 'Ready!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 2, cv2.LINE_AA)
                            # Set tracking_started on next frame
                            if elapsed >= ready_hold_duration + 0.5:  # small delay so Ready! is visible
                                tracking_started = True
                        else:
                            # Show hold message with countdown
                            seconds_left = int(ready_hold_duration - elapsed + 1)
                            cv2.putText(image, f'Hold still... {seconds_left}s',
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, cv2.LINE_AA)
                    else:
                        if not tracking_started:
                            ready_start_time = None


                    # Show feedback message to user
                    # Show feedback message to user
                    if not tracking_started:
                        if is_ready:
                            elapsed = time.time() - ready_start_time if ready_start_time else 0
                            if elapsed >= ready_hold_duration:
                                cv2.putText(image, 'Ready!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 255, 0), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(image, 'Hold still... Getting Ready',
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, cv2.LINE_AA)

                    # Track only after ready
                    if tracking_started:
                        if current_exercise == 'bicep_curl':
                            track_bicep_curls(landmarks)
                        elif current_exercise == 'squat':
                            track_squats(landmarks)
                        elif current_exercise == 'bench_press':
                            track_bench_press(landmarks)
                        elif current_exercise == 'deadlift':
                            track_deadlift(landmarks)
                        elif current_exercise == 'shoulder_press':
                            track_shoulder_press(landmarks)




            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reps')
def get_reps():
    return jsonify({
        'left': counter_left,
        'right': counter_right
    })

@app.route('/reset', methods=['POST'])
def reset_reps():
    global counter_left, counter_right
    counter_left = 0
    counter_right = 0
    return jsonify({'message': 'Rep counts reset successfully'}), 200

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracking_enabled
    tracking_enabled = True
    return jsonify({'message': 'Tracking started'}), 200

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_enabled
    tracking_enabled = False
    return jsonify({'message': 'Tracking stopped'}), 200

@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    global current_exercise
    data = request.get_json()
    exercise = data.get('exercise')
    if exercise in ['bicep_curl', 'squat', 'bench_press', 'deadlift', 'shoulder_press']:
        current_exercise = exercise
        return jsonify({'message': f'Set exercise to {exercise}'}), 200
    return jsonify({'error': 'Invalid exercise'}), 400

@app.route('/reset_ready', methods=['POST'])
def reset_ready():
    global ready_start_time, tracking_started
    ready_start_time = None
    tracking_started = False
    return jsonify({'message': 'Ready timer reset'}), 200

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global counter_left, counter_right, tracking_enabled, tracking_started, ready_start_time

    if 'frame' not in request.files:
        return jsonify({'error': 'No frame uploaded'}), 400

    file = request.files['frame']
    exercise = request.form.get('exercise', 'bicep_curl')

    # Read image from bytes
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid image data'}), 400

    ready_state = "not_ready"

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Ready check
            ready_func = {
                'bicep_curl': is_ready_bicep_curl,
                'squat': is_ready_squat,
                'bench_press': is_ready_bench_press,
                'deadlift': is_ready_deadlift,
                'shoulder_press': is_ready_shoulder_press,
            }.get(exercise, lambda _: False)

            if tracking_enabled:
                if ready_func(landmarks):
                    if ready_start_time is None:
                        ready_start_time = time.time()
                        ready_state = "holding"
                    elif time.time() - ready_start_time >= ready_hold_duration:
                        tracking_started = True
                        ready_state = "ready"
                    else:
                        ready_state = "holding"
                else:
                    if not tracking_started:
                        ready_start_time = None
                        ready_state = "not_ready"

                if tracking_started:
                    tracker_func = {
                        'bicep_curl': track_bicep_curls,
                        'squat': track_squats,
                        'bench_press': track_bench_press,
                        'deadlift': track_deadlift,
                        'shoulder_press': track_shoulder_press,
                    }.get(exercise)
                    if tracker_func:
                        tracker_func(landmarks)

    return jsonify({
        'left': counter_left,
        'right': counter_right,
        'ready_state': ready_state
    })





if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
