import cv2
from tqdm import tqdm
import numpy as np
from collections import Counter
# NUM_PIXELS = 150

class point:
    def __init__(self, x, y=None, z=None):
        if y is None and z is None: 
            self.x = x
            self.y = x
            self.z = x
            self.xyz = (x, x, x)
        else:  
            self.x = x
            self.y = y
            self.z = z
            self.xyz = (x, y, z)
    
    def fixY(self, newY):
        return point(self.x, newY, self.z)
    
    def __add__(self, other):
        return point(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return point(self.x - other.x, self.y - other.y, self.z - other.z)
        

    def __call__(self):
        return self.xyx

class lithograph:
    def __init__(self, image):
        img = cv2.imread(image)

        img, palette = self.smooth_colors(img)
        self.save_palette(palette)
        self.image = self.readImage(img)
        self.stl = "solid\n"
        self.facets = []
        # need to add endsolid
    
    # forces all pixels to color it is closest to
    def smooth_colors(self, img, n_colors=9, min_dist=50):
        """
        Load an image, find its n most frequent colors, and build a palette
        of n colors such that each is at least `min_dist` away from each other.
        Then snap every pixel to the nearest palette color.

        Parameters
        ----------
        img_path : str
            Path to an RGB image file.
        n_colors : int
            Desired number of colors in the palette.
        min_dist : float
            Minimum Euclidean distance between any two palette colors.

        Returns
        -------
        quantized : np.ndarray
            Quantized image array of shape (H, W, 3), dtype uint8.
        palette : np.ndarray
            Array of shape (n_colors, 3) of the chosen RGB colors.
        """

        # 1) load and convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2) count frequencies
        pixels = img.reshape(-1, 3)
        counts = Counter(map(tuple, pixels))

        # 3) iterate over the most common colors and pick only those far enough apart
        palette = []
        for color, _ in counts.most_common():
            c = np.array(color, dtype=int)
            if not palette:
                palette.append(c)
            else:
                # compute distance to existing palette
                dists = np.linalg.norm(np.stack(palette) - c, axis=1)
                if np.all(dists >= min_dist):
                    palette.append(c)
            if len(palette) >= n_colors:
                break

        # If we didn't get enough, pad with next most common ignoring distance
        if len(palette) < n_colors:
            for color, _ in counts.most_common():
                c = np.array(color, dtype=int)
                if not any((c == p).all() for p in palette):
                    palette.append(c)
                if len(palette) >= n_colors:
                    break

        palette = np.stack(palette).astype(np.uint8)  # shape (n_colors,3)

        # 4) assign each pixel to nearest palette color
        flat = pixels.astype(int)
        diffs = flat[:, None, :] - palette[None, :, :].astype(int)  # shape (num_pixels, n_colors, 3)
        dists = np.linalg.norm(diffs, axis=2)                      # shape (num_pixels, n_colors)
        idx = np.argmin(dists, axis=1)                            # shape (num_pixels,)

        quantized = palette[idx].reshape(img.shape)

        return quantized, palette


    def save_palette(self, palette, filename="palette.png", block_size=50):
        """
        Save a color palette as an image.

        Parameters
        ----------
        palette : array-like of shape (n_colors, 3)
            RGB values (0–255) of each palette color.
        filename : str
            Path to write the palette image (e.g. 'palette.png').
        block_size : int, optional
            Width and height in pixels of each color block.
        """
        palette = np.asarray(palette, dtype=np.uint8)
        n_colors = palette.shape[0]
        # Create an image of height=block_size, width=n_colors*block_size
        palette_img = np.zeros((block_size, block_size * n_colors, 3), dtype=np.uint8)

        for i, color in enumerate(palette):
            start = i * block_size
            end   = start + block_size
            # fill the block with this RGB color
            palette_img[:, start:end, :] = color

        # OpenCV expects BGR ordering
        bgr = cv2.cvtColor(palette_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr)

    
    def readImage(self, img, dark_high=False):
        greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        greyImg = cv2.resize(greyImg, (0, 0), fx=1, fy=1)
        greyImg = cv2.GaussianBlur(greyImg, (5,5), 2.0)
        greyImg = cv2.flip(greyImg, 1)
        
        if dark_high:
            image = 255 - greyImg
        cv2.imwrite("mod.png", image)

        return image
    
    def addTriangle(self, p1, p2, p3):
        triangle = "facet normal 0.0 0.0 0.0\n"
        triangle += "outer loop\n"
        
        triangle += f'vertex {p1.x} {p1.y} {p1.z}\n'
        triangle += f'vertex {p2.x} {p2.y} {p2.z}\n'
        triangle += f'vertex {p3.x} {p3.y} {p3.z}\n'

        
        triangle += "endloop\n"
        triangle += "endfacet\n"
        # self.stl += triangle
        self.facets.append(triangle)
        
    def addRect(self, p1, p2, p3, p4, invert=False):
        
        if invert:
            temp = p2
            p2 = p3
            p3 = temp
        self.addTriangle(p1, p2, p3)
        self.addTriangle(p3, p2, p4)    
    
    def addPixelSides(self, bottomLeft, y, scale=1):
        blz = bottomLeft.fixY(0)
        across = blz + point(scale, 0, scale)
        self.addRect(blz, blz + point(0, 0, scale), bottomLeft.fixY(y), bottomLeft.fixY(y) + point(0, 0, scale))
        self.addRect(blz, blz + point(scale, 0, 0), bottomLeft.fixY(y), bottomLeft.fixY(y) + point(scale, 0, 0), invert=True)
        
        self.addRect(across, across - point(0, 0, scale), across.fixY(y), across.fixY(y) - point(0, 0, scale))
        self.addRect(across, across - point(scale, 0, 0), across.fixY(y), across.fixY(y) - point(scale, 0, 0), invert=True)
        
    
    
    def add2DPixel(self, bottomLeft, y, invert=False, scale=1):
        self.addRect(bottomLeft.fixY(y), bottomLeft + point(0, y, scale), bottomLeft + point(scale, y, 0), bottomLeft + point(scale, y, scale), invert)
    
    def add3Dpixel(self, bottomLeft, height, scale=1):
        self.add2DPixel(bottomLeft.fixY(0), 0, scale=scale)
        self.add2DPixel(bottomLeft, height, scale=scale)
        self.addPixelSides(bottomLeft, height, scale=scale)
    
    def addCuboid(self, bottomLeft, width, depth, height):
        """
        Draw a rectangular prism with:
          - one corner at `bottomLeft` (a point),
          - `width` along +x,
          - `depth` along +z,
          - `height` along +y.
        """
        p000 = bottomLeft
        p100 = p000 + point(width, 0, 0)
        p001 = p000 + point(0, 0, depth)
        p101 = p000 + point(width, 0, depth)
        p010 = p000 + point(0, height, 0)
        p110 = p000 + point(width, height, 0)
        p011 = p000 + point(0, height, depth)
        p111 = p000 + point(width, height, depth)

        # bottom face
        self.addRect(p000, p001, p100, p101)
        # top face (flip winding)
        self.addRect(p010, p110, p011, p111, invert=True)
        # four side faces
        self.addRect(p000, p010, p001, p011)           # x=0 side
        self.addRect(p100, p101, p110, p111, invert=True)  # x=width side
        self.addRect(p000, p100, p010, p110, invert=True)  # z=0 side
        self.addRect(p001, p011, p101, p111)           # z=depth side
    
    def magic(self, scale=1, levels=30, plate_height=3):
        img     = self.image
        n_rows  = len(img)
        n_cols  = len(img[0])
        top     = max(max(row) for row in img)

        # 1) Draw the uniform plate underneath:
        self.addCuboid(
            bottomLeft = point(0, 0, 0),
            width      = n_rows * scale,
            depth      = n_cols * scale,
            height     = plate_height
        )

        # 2) Precompute the “above‐plate” heights for each distinct pixel value
        unit = (top // levels) / 85.0
        height_above = {}
        for v in {v for row in img for v in row}:
            layer = (v * levels) // top if v > 0 else 0
            height_above[v] = layer * unit

        # 3) For each row, run‐length encode contiguous same‐height runs
        for i in tqdm(range(n_rows)):
            row_vals   = img[i]
            row_heights = [height_above[v] for v in row_vals]

            j = 0
            while j < n_cols:
                h = row_heights[j]
                # skip flat regions (just plate)
                if h == 0:
                    j += 1
                    continue

                start = j
                while j < n_cols and row_heights[j] == h:
                    j += 1
                length = j - start

                # draw one cuboid sitting on top of the plate
                bottom = point(i * scale, plate_height, start * scale)
                self.addCuboid(
                    bottomLeft = bottom,
                    width      = scale,
                    depth      = length * scale,
                    height     = h
                )
    
    def saveStl(self, filename):
        f = open(f'{filename}.stl', "w")
        out = self.stl + "endsolid\n"
        f.write(out)
        f.close()
    
    def saveStlFromArray(self, filename):
        print("writing")
        with open(f"{filename}.stl", "w") as txt_file:
            txt_file.write("solid\n")
            for line in self.facets:
                txt_file.write(line)
            txt_file.write("endsolid\n")


def main():
    # myLitho = lithograph("calder_test.jpeg")
    myLitho = lithograph("poster2.tif")
    
    
    # myLitho.addRect(point(0, 0, 0), point(0, 0, 1),  point(1, 0, 0), point(1, 0, 1))
    # myLitho.add3Dpixel(point(0), 7)
    myLitho.magic(1, 10)
    myLitho.saveStlFromArray("poster2")
    # myLitho.readImage("jim.png")
    
    
    
if __name__ == "__main__":
    main()