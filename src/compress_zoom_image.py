import cv2
import os

input_path = r"c:\Users\Falker\Desktop\Code\KT\paper\optical_aliasing_zoom.png"
output_path = r"c:\Users\Falker\Desktop\Code\KT\paper\optical_aliasing_zoom_optimized.png"

def compress_image(source, dest):
    img = cv2.imread(source)
    if img is None:
        print("Source file not found or cannot be read")
        return
    
    # Save as optimized PNG
    cv2.imwrite(dest, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    # Save as high-quality JPEG
    jpeg_dest = dest.replace(".png", ".jpg")
    cv2.imwrite(jpeg_dest, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

if __name__ == "__main__":
    if os.path.exists(input_path):
        compress_image(input_path, output_path)
        print(f"Original size: {os.path.getsize(input_path)} bytes")
        if os.path.exists(output_path):
            print(f"Optimized PNG size: {os.path.getsize(output_path)} bytes")
        if os.path.exists(output_path.replace('.png', '.jpg')):
            print(f"Optimized JPG size: {os.path.getsize(output_path.replace('.png', '.jpg'))} bytes")
    else:
        print("Source file not found")
