import cv2
import mediapipe as mp
import pygame



# Initialize pygame mixer for sounds
pygame.mixer.init()
snare_sound = pygame.mixer.Sound("snare.wav")
hihat_sound = pygame.mixer.Sound("hihat.wav")
kick_sound = pygame.mixer.Sound("kick.wav")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

#rcb
#hehe

# To avoid continuous sound while finger stays inside zone
played = {"snare": False, "hihat": False, "kick": False}

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, c = frame.shape

    # Draw drum zones
    #cv2.rectangle(frame, (50, 50), (250, 250), (255, 0, 0), 2)   # Snare zone
    cv2.rectangle(frame, (390, 50), (590, 250), (0, 255, 0), 2)  # Hi-hat zone
    cv2.rectangle(frame, (220, 300), (420, 480), (0, 0, 255), 2) # Kick zone

    # Snare zone (bottom-left)
    cv2.rectangle(frame, (50, 300), (200, 460), (255, 0, 0), 2)

# Hi-hat zone (bottom-center)
    #cv2.rectangle(frame, (250, 300), (400, 460), (0, 255, 0), 2)

# Kick zone (bottom-right)
    #cv2.rectangle(frame, (450, 300), (600, 460), (0, 0, 255), 2)


    # Reset "played" when finger leaves zones
    in_zone = {"snare": False, "hihat": False, "kick": False}

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                px, py = int(lm.x * w), int(lm.y * h)

                if id == 8:  # Index fingertip
                    cv2.circle(frame, (px, py), 15, (0, 255, 0), cv2.FILLED)

                    if 50 < px < 200 and 300 < py < 460:
                        cv2.putText(frame, "Snare!", (px, py-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                        in_zone["snare"] = True
                        if not played["snare"]:
                            snare_sound.play()
                            played["snare"] = True

                    elif 390 < px < 590 and 50 < py < 250:
                        cv2.putText(frame, "Hi-Hat!", (px, py-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        in_zone["hihat"] = True
                        if not played["hihat"]:
                            hihat_sound.play()
                            played["hihat"] = True

                    elif 220 < px < 420 and 300 < py < 480:
                        cv2.putText(frame, "Kick!", (px, py-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        in_zone["kick"] = True
                        if not played["kick"]:
                            kick_sound.play()
                            played["kick"] = True

    # Reset zones when finger leaves
    for zone in played:
        if not in_zone[zone]:
            played[zone] = False

    cv2.imshow("Virtual Drum Kit", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()





"""import cv2                          # OpenCV: read webcam frames, draw, show windows.
import mediapipe as mp              # MediaPipe: Google’s hand-tracking library.

mp_hands = mp.solutions.hands       # Shortcut to the Hands solution module.
hands = mp_hands.Hands(             # Create a real-time hand detector/tracker.
    min_detection_confidence=0.7,   # 0–1: how sure the model must be to detect a new hand.
    min_tracking_confidence=0.7     # 0–1: how sure it must be to keep tracking between frames.
)
mp_draw = mp.solutions.drawing_utils  # Helpers to draw landmarks & connections.

cap = cv2.VideoCapture(0)           # Open the default camera (index 0).

while cap.isOpened():               # Loop as long as the camera is open/working.
    ret, frame = cap.read()         # Grab a frame; ret=True if success, frame is the image (BGR).
    frame = cv2.flip(frame, 1)      # Mirror horizontally so it feels like a mirror.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR→RGB (MediaPipe expects RGB).
    result = hands.process(rgb)     # Run the hand model → returns landmarks if hands are found.

    if result.multi_hand_landmarks:                       # If any hands were detected…
        for handLms in result.multi_hand_landmarks:       # For each detected hand…
            mp_draw.draw_landmarks(                       # Draw points + skeleton on the frame.
                frame,                                    # Image to draw on (BGR).
                handLms,                                  # 21 landmark points (normalized coords).
                mp_hands.HAND_CONNECTIONS                 # Which landmarks to connect with lines.
            )

    cv2.imshow("Hand Tracking", frame)  # Show the annotated frame in a window.
    if cv2.waitKey(1) & 0xFF == 27:     # Check for a key every ~1ms; 27 = Esc key → quit.
        break                           # Leave the loop (clean shutdown).

cap.release()                    # Free the camera device.
cv2.destroyAllWindows()          # Close any OpenCV windows that were opened.
"""
