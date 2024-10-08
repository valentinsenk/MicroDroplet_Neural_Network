import os
import argparse
from PIL import Image, ImageDraw

def create_collage(image_dir, output_path, grid_size=20, image_size=(512, 512), zoom_image_size=(1024, 512)):
    # Create a blank canvas for the collage with 20 columns (because each row will have 2 images per subdir)
    collage_width = grid_size * image_size[0]  # Each row will fit two images per subdir (1 regular, 1 zoom)
    collage_height = (len(os.listdir(image_dir)) // (grid_size // 2) + 1) * image_size[1]  # Adjust height based on image count
    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))  # White background

    # Get all the subdirectories and sort them
    subdirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])

    # Add each image to the collage
    for index, subdir in enumerate(subdirs):
        # Compute the row and column for the current image
        row = index // (grid_size // 2)  # Two images per subdir, so each subdir takes two columns
        col = (index % (grid_size // 2)) * 2  # Each subdir uses 2 columns (1 for mIFSS-Disp.png, 1 for the zoom image)

        # Path to mIFSS-Disp.png
        disp_image_path = os.path.join(image_dir, subdir, f"lhs_{subdir}_v1_results", "mIFSS-Disp.png")
        # Path to 03_View_ZOOM_Deformed.png
        zoom_image_path = os.path.join(image_dir, subdir, f"lhs_{subdir}_v1_results", "pics_and_videos", "03_View_ZOOM_Deformed.png")

        # Handle mIFSS-Disp.png
        if os.path.exists(disp_image_path):
            disp_img = Image.open(disp_image_path).resize(image_size)
        else:
            # Create a blank placeholder if no image is found
            disp_img = Image.new('RGB', image_size, color=(255, 255, 255))  # White blank image
            draw = ImageDraw.Draw(disp_img)
            draw.text((10, 10), "No Image", fill=(0, 0, 0))  # Add "No Image" text to the blank image

        # Paste the mIFSS-Disp.png in the correct position
        collage_image.paste(disp_img, (col * image_size[0], row * image_size[1]))

        # Handle 03_View_ZOOM_Deformed.png
        if os.path.exists(zoom_image_path):
            zoom_img = Image.open(zoom_image_path).resize(zoom_image_size)  # Double the width
        else:
            # Create a blank placeholder if no image is found
            zoom_img = Image.new('RGB', zoom_image_size, color=(255, 255, 255))  # White blank image
            draw = ImageDraw.Draw(zoom_img)
            draw.text((10, 10), "No Image", fill=(0, 0, 0))  # Add "No Image" text to the blank image

        # Paste the 03_View_ZOOM_Deformed.png next to mIFSS-Disp.png
        collage_image.paste(zoom_img, ((col + 1) * image_size[0], row * image_size[1]))

    # Save the final collage
    collage_image.save(output_path, "PNG")
    print(f"Collage saved to {output_path}")

# Main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a collage from PNG images')
    parser.add_argument('--root_dir', required=True, help='Root directory containing subdirectories with PNG files')
    parser.add_argument('--output', required=True, help='Output file for the collage')
    args = parser.parse_args()

    # Create the collage
    create_collage(args.root_dir, args.output, grid_size=20, image_size=(512, 512), zoom_image_size=(1024, 512))
