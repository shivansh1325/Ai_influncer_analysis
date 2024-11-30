import pandas as pd
import numpy as np
import requests
import os
import hashlib
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Haar Cascade for face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_faces_from_video(video_url, output_dir='faces', every_nth_frame=5):
    """
    Download and process a video to extract all unique faces across all frames.
    Returns a list of face encodings and the corresponding filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    face_encodings = []
    face_filenames = []
    log_errors = []

    try:
        video_filename = os.path.join(output_dir, f"{hashlib.md5(video_url.encode()).hexdigest()}.mp4")

        response = requests.get(video_url, stream=True)
        with open(video_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        cap = cv2.VideoCapture(video_filename)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % every_nth_frame != 0:
                continue

            # Detected faces
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in face_locations:
                face = gray_frame[y:y+h, x:x+w]
                
                resized_face = cv2.resize(face, (64, 64))
                face_encoding = resized_face.flatten()  # Flatten into a fixed-length vector

                match = False
                for known_enc in face_encodings:
                    if cosine_similarity([known_enc], [face_encoding])[0][0] > 0.9:  # Similarity threshold
                        match = True
                        break

                if not match:
                    # Saveing the face encoding and image
                    face_encodings.append(face_encoding)
                    face_filename = os.path.join(output_dir, f"{hashlib.md5(face_encoding.tobytes()).hexdigest()}.jpg")
                    face_filenames.append(face_filename)
                    cv2.imwrite(face_filename, frame[y:y+h, x:x+w])

        cap.release()

    except Exception as e:
        log_errors.append(f"Error processing video {video_url}: {e}")

    return face_encodings, face_filenames, log_errors


def analyze_influencer_performance(csv_path, output_csv='influencer_performance.csv', output_log='error_log.txt'):
    """
    Identify unique influencers, calculate performance, and analyze engagement.
    """
    df = pd.read_csv(csv_path)
    unique_videos = df['Video URL'].unique()
    influencer_data = {}
    error_log = []

    for video_url in unique_videos:
        print(f"Processing video: {video_url}")
        face_encodings, face_filenames, log_errors = extract_faces_from_video(video_url)
        error_log.extend(log_errors)

        if not face_encodings:
            print(f"No faces detected in video: {video_url}")
            continue

        video_performances = df[df['Video URL'] == video_url]['Performance']
        avg_performance = video_performances.mean()

        for face_enc, face_filename in zip(face_encodings, face_filenames):
            match_found = False
            for influencer_id, data in influencer_data.items():
                if len(data['face_encoding']) == len(face_enc):
                    if cosine_similarity([data['face_encoding']], [face_enc])[0][0] > 0.9:  # Match threshold
                        influencer_data[influencer_id]['video_count'] += 1
                        influencer_data[influencer_id]['performance_scores'].append(avg_performance)
                        influencer_data[influencer_id]['face_images'].append(face_filename)
                        match_found = True
                        break

            if not match_found:
                influencer_id = f"influencer_{len(influencer_data) + 1}"
                influencer_data[influencer_id] = {
                    'face_encoding': face_enc,
                    'video_count': 1,
                    'performance_scores': [avg_performance],
                    'face_images': [face_filename]
                }

    influencer_analysis = []
    for influencer_id, data in influencer_data.items():
        avg_perf = np.mean(data['performance_scores'])
        perf_variance = np.var(data['performance_scores'])

        influencer_analysis.append({
            'Influencer ID': influencer_id,
            'Average Performance': avg_perf,
            'Performance Variance': perf_variance,
            'Video Count': data['video_count'],
            'Face Images': ", ".join(data['face_images'])
        })

    influencer_df = pd.DataFrame(influencer_analysis)
    influencer_df = influencer_df.sort_values(by=['Average Performance', 'Performance Variance'], ascending=[False, True])
    influencer_df.to_csv(output_csv, index=False)

    with open(output_log, 'w') as log_file:
        log_file.write("\n".join(error_log))

    return influencer_df
# Main Execution
if __name__ == "__main__":
    try:
        input_csv = '/content/Assignment Data - Sheet1 (1).csv'
        results = analyze_influencer_performance(input_csv)
        print("Influencer analysis completed. Results saved to 'influencer_performance.csv'")
        print(results)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
