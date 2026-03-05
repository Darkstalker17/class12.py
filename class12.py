import cv2, time, pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.vision import hand_landmarker, hand_landmarker_options

# Config
SCROLL_SPEED = 300
SCROLL_DELAY = 1
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Initialize Hand Landmarker (new API)
options = hand_landmarker_options.HandLandmarkerOptions(
    max_num_hands=1
)
landmarker = hand_landmarker.HandLandmarker.create_from_options(options)

def detect_gesture(hand_landmarks):
    """Determine gesture: all fingers up = scroll up, fist = scroll down"""
    tips_y = [hand_landmarks.landmarks[i].y for i in [
        8, 12, 16, 20  # INDEX, MIDDLE, RING, PINKY tips
    ]]
    pip_y = [hand_landmarks.landmarks[i].y for i in [
        6, 10, 14, 18  # corresponding PIP joints
    ]]

    fingers_up = sum([1 if tip < pip else 0 for tip, pip in zip(tips_y, pip_y)])

    # Thumb
    thumb_tip = hand_landmarks.landmarks[4]
    thumb_ip = hand_landmarks.landmarks[3]
    fingers_up += 1 if thumb_tip.x > thumb_ip.x else 0

    return "scroll_up" if fingers_up == 5 else "scroll_down" if fingers_up == 0 else "none"

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
last_scroll = p_time = 0

print("Gesture Scroll Control Active\nOpen palm: Scroll Up\nFist: Scroll Down\nPress 'q' to exit")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    frame = python.vision.Image(image_format=python.vision.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(frame)

    gesture = "none"
    if result.hands:
        hand_landmarks = result.hands[0]
        gesture = detect_gesture(hand_landmarks)

        if (time.time() - last_scroll) > SCROLL_DELAY:
            if gesture == "scroll_up":
                pyautogui.scroll(SCROLL_SPEED)
            elif gesture == "scroll_down":
                pyautogui.scroll(-SCROLL_SPEED)
            last_scroll = time.time()

    fps = 1 / (time.time() - p_time) if (time.time() - p_time) > 0 else 0
    p_time = time.time()
    cv2.putText(img, f"FPS: {int(fps)} | Gesture: {gesture}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()