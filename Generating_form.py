import pandas as pd
import os
from IPython.display import HTML

def generate_table_from_local_images(csv_path, faces_dir='faces', output_html='influencer_table.html'):
    
    df = pd.read_csv(csv_path)

    available_images = set(os.listdir(faces_dir))

    def get_local_image(face_images):
        
        for img_path in face_images.split(', '):
            img_name = os.path.basename(img_path)  # Extract the filename
            if img_name in available_images:
                return f"<img src='{faces_dir}/{img_name}' width='100' height='100' alt='Face'>"
        return "Image Not Found"

    df['Face Image'] = df['Face Images'].apply(get_local_image)

    display_df = df[['Face Image', 'Influencer ID', 'Average Performance']]

    html_table = display_df.to_html(escape=False, index=False)

    with open(output_html, 'w') as f:
        f.write(html_table)

    print(f"Table generated and saved to {output_html}")
    return HTML(html_table)


# Main Execution
if __name__ == "__main__":
    try:
        table = generate_table_from_local_images(
            csv_path='influencer_performance_combined.csv', 
            faces_dir='faces'
        )
        display(table)
    except Exception as e:
        print(f"An error occurred: {e}")
