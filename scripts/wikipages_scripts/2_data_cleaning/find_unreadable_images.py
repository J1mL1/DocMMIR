import os
import traceback

from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError


def find_unreadable_images(directory, size_limit_mb=10):
    truncated_count = 0
    unreadable_files = []
    total_files = 0
    unreadable_count = 0
    bomb_count = 0
    other_error = 0
    value_error = 0
    os_error = 0
    large_files = []  # List to store large files

    size_limit_bytes = size_limit_mb * 1024 * 1024  # Convert MB to bytes

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            total_files += 1  # Increment total files count
            file_path = os.path.join(root, filename)

            # Check the file size
            try:
                file_size = os.path.getsize(file_path)
                if file_size > size_limit_bytes:
                    print(f"Large file detected (> {size_limit_mb} MB): {file_path} ({file_size} bytes)")
                    large_files.append(file_path)
                    continue  # Skip processing this file
            except OSError as e:
                print(f"OS error when checking size of {file_path}: {e}")
                unreadable_files.append(file_path)
                os_error += 1
                continue

            try:
                # Attempt to open the image file
                with Image.open(file_path) as img:
                    img.verify()  # Verify if it's a readable image

            except (UnidentifiedImageError, IOError):
                # If an error occurs, add file to unreadable list
                unreadable_files.append(file_path)
                unreadable_count += 1  # Increment unreadable count
                print(f"Unreadable image file: {file_path}")

            except DecompressionBombError:
                # Skip the file if it's too large (decompression bomb)
                print(f"Skipping large image (decompression bomb): {file_path}")
                unreadable_files.append(file_path)
                bomb_count += 1

            except OSError as e:
                # Handle specific OSError cases like Truncated File
                if "Truncated File" in str(e):
                    print(f"Truncated file error: {file_path}")
                    truncated_count += 1
                else:
                    print(f"Other OSError: {e} | File: {file_path}")
                    os_error += 1
                unreadable_files.append(file_path)

            except ValueError as e:
                print(f"Value error: {e} | File: {file_path}")
                unreadable_files.append(file_path)
                value_error += 1

            except Exception as e:
                # Catch-all for unexpected errors
                print(f"Unexpected error with file: {file_path}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {e}")
                traceback.print_exc()  # Print the full traceback for debugging
                unreadable_files.append(file_path)

    print(f"\nTotal files examined: {total_files}")
    print(f"Total unreadable files: {unreadable_count}")
    print(f"Large files skipped: {len(large_files)}")
    print(f"Decompression bomb files: {bomb_count}")
    print(f"Truncated files: {truncated_count}")
    print(f"Value errors: {value_error}")
    print(f"OS errors: {os_error}")
    print(f"Other errors: {other_error}")

    return unreadable_files, large_files


if __name__ == "__main__":
    directory_path = '/mnt/data1_HDD_14TB/docmmir/images/'

    base_dir = os.path.expanduser("~/zirui/DocMMIR/data")
    if not os.path.exists(base_dir):
        print("Output directory does not exist. It will be created.")
    else:
        print("Output directory exists.")
    os.makedirs(base_dir, exist_ok=True)

    # Find unreadable images and large files
    unreadable_files, large_files = find_unreadable_images(directory_path)

    # Save unreadable files to a text file
    unreadable_files_path = os.path.join(base_dir, "unreadable_files.txt")
    with open(unreadable_files_path, "w") as f:
        for file_path in unreadable_files:
            f.write(file_path + "\n")

    # Save large files to a separate text file
    large_files_path = os.path.join(base_dir, "large_files.txt")
    with open(large_files_path, "w") as f:
        for file_path in large_files:
            f.write(file_path + "\n")

    print("\nUnreadable files saved to 'unreadable_files.txt'")
    print("Large files saved to 'large_files.txt'")
