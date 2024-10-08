import os
import argparse
from PIL import Image, ImageDraw

def create_collage(image_dir, output_path, grid_size=(10, 10), image_size=(256, 256)):
    # Create a blank canvas for the collage
    collage_width = grid_size[0] * image_size[0]
    collage_height = grid_size[1] * image_size[1]
    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))  # White background

    # Get all the subdirectories and sort them
    subdirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])

    # Add each image to the collage
    for index, subdir in enumerate(subdirs):
        # Compute the row and column for the current image
        row = index // grid_size[0]
        col = index % grid_size[0]

        # Construct the image path (e.g., '001/lhs_001_v1_results/mIFSS-Disp.png')
        image_path = os.path.join(image_dir, subdir, f"lhs_{subdir}_v1_results", "mIFSS-Disp.png")
        
        if os.path.exists(image_path):
            img = Image.open(image_path).resize(image_size)
        else:
            # Create a blank placeholder if no image is found
            img = Image.new('RGB', image_size, color=(255, 255, 255))  # White blank image
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "No Image", fill=(0, 0, 0))  # Add "No Image" text to the blank image

        # Paste the image (or placeholder) in the correct position
        collage_image.paste(img, (col * image_size[0], row * image_size[1]))

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
    create_collage(args.root_dir, args.output, grid_size=(10, 10), image_size=(256, 256))
