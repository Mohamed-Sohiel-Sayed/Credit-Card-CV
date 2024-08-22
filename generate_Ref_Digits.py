from PIL import Image, ImageDraw, ImageFont
import os

IMAGE_SIZE=(50,70)
FONT_DIRECTORY="font/CREDC___.ttf"              # font file path
FONT_SIZE= 40                                   # in pixels


def createDigitImage(num:int,font,directory):
    image = Image.new('L', IMAGE_SIZE, color=0)  # Create a black image, 'L' (8-bit pixels, grayscale), 
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), str(num), font=font, fill=255)  # Draw the digit
    image.save(f'{directory}/template_{num}.jpg')  # Save the image


def ensure_directory_and_files(directory,files,font):
    if not os.path.exists(directory):       # ensure directory exists
            os.makedirs(directory)

        # Check for each file in the directory
    for i,file_name in enumerate(files):
        file_path = os.path.join(directory, file_name)
        
        if not os.path.exists(file_path):       # if files doesn't exist, create it
            createDigitImage(i,font,directory)

def run():
    directory = 'Ref_Digits'
    files = [f"template_{i}" for i in range(10)]

    # Ensure the directory and files are in place
    font = ImageFont.truetype(FONT_DIRECTORY, FONT_SIZE)  # Specify the font and size
    ensure_directory_and_files(directory, files,font)
    return [f"{directory}/{file}.jpg" for file in files]


if __name__=="__main__":
    run()