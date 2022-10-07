import collections
import sys
import os
import time
import cv2
import face_recognition
import argparse

def load_people():
    encodings, names = [], []
    for file in os.listdir('people'):
        if file.endswith('.jpeg'):
            img = face_recognition.load_image_file(f'people/{file}')

            encodings.append(face_recognition.face_encodings(img)[0])

            # Remove the file extension
            name = file.split('.')[0]

            # Split the name by underscores
            name = name.split("_")

            # Capitalize name
            name = " ".join([n.capitalize() for n in name])
            names.append(name)

    return encodings, names

def find_people(people, frame):
    # Make the image smaller to speed up processing
    scale = 4
    frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)

    # Obtain locations of all faces in frame
    face_locations = face_recognition.face_locations(frame, model="hog")
    face_encodings = face_recognition.face_encodings(frame, face_locations, model="small")

    # Results
    processed_idxs = set()
    results = []

    for encoding, name in zip(*people):
        matches = face_recognition.compare_faces(face_encodings, encoding, tolerance=0.45)

        for idx, val in enumerate(matches):
            if val:
                # Scale back up face locations since the frame we detected in was scaled to 1/size
                r = tuple(f * scale for f in face_locations[idx])

                # Remove the encoding since we don't need it anymore
                processed_idxs.add(idx)

                # Add to results
                results.append((r, name))

    # Process remaining faces as Unknown
    for idx in range(len(face_encodings)):
        if idx in processed_idxs:
            continue

        r = tuple(f * scale for f in face_locations[idx])
        results.append((r, "Unknown"))

    return results

def find_and_display(people):
    cap = cv2.VideoCapture(0)
    frame_num = 0
    finds = []
    prev_time = time.time()

    # Settings
    font = cv2.FONT_HERSHEY_DUPLEX
    process_every = 4
    success_color = (0, 255, 0)
    fail_color = (0, 0, 255)
    text_color = (255, 255, 255)

    # FPS
    fps = 0
    fps_last_x = collections.deque(maxlen=process_every)

    def update_fps():
        nonlocal fps, prev_time

        now = time.time()
        took = now - prev_time
        prev_time = now
        if fps_last_x.maxlen == len(fps_last_x):
            fps_last_x.popleft()
        fps_last_x.append(1 / took)

        # Calculate average of last x frames
        if len(fps_last_x) > 0:
            fps = sum(fps_last_x) / len(fps_last_x)


    while True:
        # Read frame from camera
        _, frame = cap.read()

        # Process every Xth frame
        if frame_num % process_every == 0:
            finds = find_people(people, frame)
        
        # Draw boxes around faces and names
        for res in finds:
            (top, right, bottom, left), name= res

            color = success_color if name != "Unknown" else fail_color

            cv2.rectangle(frame, (left, top), (right, bottom), color, 8)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, text_color, 2)

        # Display FPS
        update_fps()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1, text_color, 2)

        # Display frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        frame_num += 1
    
    cap.release()
    cv2.destroyAllWindows()

def find_and_print(people):
    cap = cv2.VideoCapture(0)

    # Read frame from camera (+ warmup)
    for _ in range(5):
        _, frame = cap.read()

    finds = find_people(people, frame)
    
    # Print names
    names = set(name for _, name in finds)
    print(f"Found: {', '.join(names)}")

    cap.release()
    cv2.destroyAllWindows()

    return names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect', action='store_true', help='Detect faces in video stream')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    people = load_people()

    if args.detect:
        find_and_display(people)
    else:
        while True:
            key = input("Press any key to find people...")
            if key == 'q': break
            find_and_print(people)
    
        