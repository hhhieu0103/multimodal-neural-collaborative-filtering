from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math

def dominant_edge_color_padding(img):
    # Resize width to 224
    w, h = img.size
    aspect = h/w
    new_w = 224
    new_h = int(new_w * aspect)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Find dominant color - focus on edge regions where background is likely
    edge_width = 5  # Sample from edges
    small_img = resized.copy()
    # Create a mask that only samples from edges
    edge_mask = Image.new('L', small_img.size, 0)
    draw = ImageDraw.Draw(edge_mask)
    # Draw white rectangles around the edges
    draw.rectangle([0, 0, small_img.width-1, edge_width], fill=255)  # Top
    draw.rectangle([0, small_img.height-edge_width, small_img.width-1, small_img.height-1], fill=255)  # Bottom
    draw.rectangle([0, 0, edge_width, small_img.height-1], fill=255)  # Left
    draw.rectangle([small_img.width-edge_width, 0, small_img.width-1, small_img.height-1], fill=255)  # Right

    # Use the mask to get only edge pixels
    edge_pixels = np.array(small_img)
    edge_mask_arr = np.array(edge_mask)
    edge_pixels = edge_pixels[edge_mask_arr == 255]

    # Find most common color in edge regions
    pixels = edge_pixels.reshape(-1, 3)
    pixel_count = {}
    for pixel in pixels:
        pixel_tuple = tuple(pixel)
        if pixel_tuple in pixel_count:
            pixel_count[pixel_tuple] += 1
        else:
            pixel_count[pixel_tuple] = 1

    # Get the most common color
    dominant_color = max(pixel_count.items(), key=lambda x: x[1])[0]

    # Pad with dominant color
    padded = Image.new("RGB", (224, 224), dominant_color)
    padded.paste(resized, (0, (224-new_h)//2))
    return padded

def reflection_padding(img):
    # Resize width to 224
    w, h = img.size
    aspect = h/w
    new_w = 224
    new_h = int(new_w * aspect)

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    resized_array = np.array(resized)

    # Calculate padding needed
    pad_top = (224 - new_h) // 2
    pad_bottom = 224 - new_h - pad_top

    # Use numpy's pad with reflection mode
    padded_array = np.pad(
        resized_array,
        ((pad_top, pad_bottom), (0, 0), (0, 0)),
        mode='reflect'
    )

    return Image.fromarray(padded_array)

def self_tiling_padding(img):
    # Resize width to 224 pixels
    w, h = img.size
    aspect = h/w
    new_w = 224
    new_h = int(new_w * aspect)

    # Resize the image
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create a new square canvas
    padded = Image.new("RGB", (224, 224))

    # Calculate how many times we need to tile vertically
    needed_height = 224
    num_tiles = math.ceil(needed_height / new_h)

    # Paste multiple copies of the image vertically
    for i in range(num_tiles):
        y_position = i * new_h
        padded.paste(resized, (0, y_position))

    # Crop to exactly 224x224
    padded = padded.crop((0, 0, 224, 224))

    return padded

def center_tiling_with_blur(img):
    # Resize width to 224 pixels
    w, h = img.size
    aspect = h/w
    new_w = 224
    new_h = int(new_w * aspect)

    # Resize the image
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create a new square canvas
    padded = Image.new("RGB", (224, 224))

    # Calculate vertical padding needed
    top_padding = (224 - new_h) // 2

    # First pass: fill the entire canvas with tiled content
    # Tile vertically both above and below the center
    for y_offset in range(0, 224, new_h):
        # Adjust to fill from top to bottom
        y_position = y_offset - (y_offset % new_h)
        padded.paste(resized, (0, y_position))

    # Second pass: paste the main content in the center (overwriting the tiled region)
    main_y_position = top_padding
    padded.paste(resized, (0, main_y_position))

    # Create a mask for the center region (where the original content is)
    mask = Image.new("L", (224, 224), 0)  # Start with black mask
    mask_draw = ImageDraw.Draw(mask)

    # Draw white rectangle for the center (original) region - fully opaque
    mask_draw.rectangle([(0, main_y_position), (224, main_y_position + new_h)], fill=255)

    # Create gradient edges for top and bottom padded regions
    fade_height = 30  # Height of the fade region

    # Top gradient (if there's padding at the top)
    if top_padding > 0:
        for y in range(fade_height):
            # Calculate alpha value (0 at the top, increasing toward the center)
            if y < top_padding:
                alpha = int(255 * y / fade_height)
                y_pos = top_padding - y
                if y_pos >= 0:
                    mask_draw.rectangle([(0, y_pos), (224, y_pos)], fill=alpha)

    # Bottom gradient (if there's padding at the bottom)
    bottom_padding_start = main_y_position + new_h
    if bottom_padding_start < 224:
        for y in range(fade_height):
            if y < (224 - bottom_padding_start):
                alpha = int(255 * (fade_height - y) / fade_height)
                y_pos = bottom_padding_start + y
                if y_pos < 224:
                    mask_draw.rectangle([(0, y_pos), (224, y_pos)], fill=alpha)

    # Apply blur to the padded regions
    # First, create a copy that will have blurred padding
    blurred = padded.copy()
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=3))

    # Final composite: use the mask to combine the original center with blurred padding
    result = Image.composite(padded, blurred, mask)

    return result