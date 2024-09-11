from PIL import Image

img = Image.open('league_result_ng.png')
img = img.convert('RGB')
pixels = img.load()

# Loop over each pixel and modify the green to blue
for i in range(img.width):
    for j in range(img.height):
        r, g, b = pixels[i, j]
        if g > b:  # If the pixel is more green than red or blue
            # Swap green and blue channels
            pixels[i, j] = (r, b, g)
        if r > g:
            pixels[i, j] = (max(r-10, 0), b, 0)

        if g > r:
            pixels[i, j] = (0, max(b-10, 0), g)

# Save the modified image
img.save('league_result_ng_b_1.png')
