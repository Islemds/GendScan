import os
import subprocess
import json
import tempfile
from PIL import Image

def extract_metadata_from_image(image):
    exiftool_path = r"C:\Users\ACER NITRO\Desktop\Data\PFE\GendScan\src\MethodeConventionnels\exiftool.exe"  # Ensure correct path

    if not os.path.exists(exiftool_path):
        print(f"Error: Exiftool not found at {exiftool_path}")
        return None

    # Create a temporary file to save the image content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        # Save the image in JPEG format
        image.save(temp_file, format='JPEG')
        temp_image_path = temp_file.name

    command = [exiftool_path, '-j', temp_image_path]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        metadata_json = result.stdout.strip()
        metadata = json.loads(metadata_json)

        # Extract relevant metadata fields and format them
        if metadata and len(metadata) > 0:
            formatted_metadata = metadata[0]  # Get the first (and only) item in the list
            display_string = "\n".join([f"**{key}**: {value}" for key, value in formatted_metadata.items()])
            return display_string

    except subprocess.TimeoutExpired:
        print("Error: Command timed out.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    finally:
        # Delete the temporary file
        os.remove(temp_image_path)

