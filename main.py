import cv2
import time
import random
import mediapipe as mp
import math
import numpy as np

Fruit_Size = 70
Spawn_Rate = 1
TOTAL_LIVES = 1e9

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def Spawn_Fruits():
    fruit = {}
    random_x = random.randint(100, 1820)
    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    fruit["Color"] = random_color
    fruit["Curr_position"] = [random_x, 700]  # Spawn from bottom
    fruit["Velocity"] = [random.uniform(-5, 5), random.uniform(-28, -36)]  # Upward
    fruit["Acceleration"] = [0, 1]  # Gravity

    Fruits.append(fruit)


def Fruit_Movement(Fruits):
    global Lives

    for fruit in Fruits[:]:  # iterate over a copy since we'll remove inside loop
        # Update velocity
        fruit["Velocity"][0] += fruit["Acceleration"][0]
        fruit["Velocity"][1] += fruit["Acceleration"][1]

        # Update position
        fruit["Curr_position"][0] += int(fruit["Velocity"][0])
        fruit["Curr_position"][1] += int(fruit["Velocity"][1])

        # Draw fruit
        cv2.circle(img, tuple(fruit["Curr_position"]), Fruit_Size, fruit["Color"], -1)

        # Remove if out of bounds
        if (fruit["Curr_position"][1] > h or fruit["Curr_position"][0] < 0 or fruit["Curr_position"][0] > w):
            Lives -= 1
            Fruits.remove(fruit)



def distance(a, b):
    x1 = a[0]
    y1 = a[1]

    x2 = b[0]
    y2 = b[1]

    d = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return int(d)

quit_game = False

w = h = 0

cap = cv2.VideoCapture(0)
_, img = cap.read()
h,w,c = img.shape

while not quit_game:
    curr_Frame = 0
    prev_Frame = 0
    delta_time = 0

    next_Time_to_Spawn = 0
    Speed = [0, 5]
    Score = 0
    Lives = TOTAL_LIVES
    game_Over = False

    slash = np.array([[]], np.int32)
    slash_Color = (255, 255, 255)
    slash_length = 19

    Fruits = []
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        success, img = cap.read()
        if not success:
            print("skipping frame")
            continue
        # print(h,w,c)
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # **************************************************************************************
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 8:
                        index_pos = (int(lm.x * w), int(lm.y * h))
                        # print("slash",slash_Color)
                        cv2.circle(img, index_pos, 18, slash_Color, -1)
                        # slash=np.delete(slash,0)
                        slash = np.append(slash, index_pos)

                        while len(slash) >= slash_length:
                            slash = np.delete(slash, len(slash) - slash_length, 0)

                        for fruit in Fruits:
                            d = distance(index_pos, fruit["Curr_position"])
                            # cv2.putText(img, str(d), fruit["Curr_position"], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, 3)
                            if (d < Fruit_Size):
                                Score = Score + 100

                                slash_Color = fruit["Color"]

                                Fruits.remove(fruit)

                # **********************************************************************************************************

        if (Lives <= 0):
            game_Over = True

        slash = slash.reshape((-1, 1, 2))
        cv2.polylines(img, [slash], False, slash_Color, 15, 0)

        curr_Frame = time.time()
        delta_Time = curr_Frame - prev_Frame
        #FPS = int(1 / delta_Time)
        #cv2.putText(img, "FPS : " + str(FPS), (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
        cv2.putText(img, "Score: " + str(Score), (int(w * 0.05), int(h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        cv2.putText(img,"Lives remaining : " + str(Lives), (int(w * 0.6), int(h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


        prev_Frame = curr_Frame

        # ***********************************************************
        if not (game_Over):
            if (time.time() > next_Time_to_Spawn):  # and not (game_Over):
                Spawn_Fruits()
                next_Time_to_Spawn = time.time() + (1 / Spawn_Rate)

            Fruit_Movement(Fruits)


        else:
            cv2.putText(img, "GAME OVER", (int(w * 0.38), int(h * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            Fruits.clear()

        #cv2.putText(img, "Lives remaining : " + str(Lives), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("img", img)

        key = cv2.waitKey(1)
        if key == ord('r') or key == ord('R'):
            break  # Restart the game loop
        elif key & 0xFF == ord("q"):
            quit_game = True
            break
    cap.release()
    cv2.destroyAllWindows()

    if quit_game:
        break  

