import pandas as pd
import numpy as np
import requests
import os
import hashlib
import cv2
import face_recognition

def extract_faces_from_video(video_url, output_dir='faces', every_nth_frame=5):
   
    os.makedirs(output_dir, exist_ok=True)
    face_encodings = []
    face_filenames = []

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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encs = face_recognition.face_encodings(rgb_frame, face_locations)

            for i, face_enc in enumerate(face_encs):
                match = False
                for known_enc in face_encodings:
                    if face_recognition.compare_faces([known_enc], face_enc, tolerance=0.6)[0]:
                        match = True
                        break

                if not match:
                    face_encodings.append(face_enc)
                    face_filename = os.path.join(output_dir, f"{hashlib.md5(face_enc.tobytes()).hexdigest()}.jpg")
                    face_filenames.append(face_filename)
                    top, right, bottom, left = face_locations[i]
                    cv2.imwrite(face_filename, frame[top:bottom, left:right])

        cap.release()

    except Exception as e:
        print(f"Error processing video {video_url}: {e}")

    return face_encodings, face_filenames


def analyze_influencer_performance(csv_path, output_csv='influencer_performance.csv'):
   
    df = pd.read_csv(csv_path)

    # Extract unique video URLs and initialize data structures
    unique_videos = df['Video URL'].unique()
    influencer_data = {}
    processed_videos = {}

    for video_url in unique_videos:
        if video_url in processed_videos:
            continue  # Skip repeated videos

        print(f"Processing video: {video_url}")

        face_encodings, face_filenames = extract_faces_from_video(video_url)

        if not face_encodings:
            print(f"No faces detected in video: {video_url}")
            continue

        video_performances = df[df['Video URL'] == video_url]['Performance']
        avg_performance = video_performances.mean()

        for face_enc, face_filename in zip(face_encodings, face_filenames):
            match_found = False
            for influencer_id, data in influencer_data.items():
                if face_recognition.compare_faces([data['face_encoding']], face_enc, tolerance=0.6)[0]:
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

    return influencer_df


# Main execution
if __name__ == "__main__":
    try:
        results = analyze_influencer_performance('/content/Assignment Data - Sheet1 (1).csv')
        print("Influencer analysis completed. Results saved to 'influencer_performance.csv'")
        print(results)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
