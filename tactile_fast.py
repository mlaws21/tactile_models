import cv2
from tqdm import tqdm
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
        self.image = self.readImage(image)
        self.stl = "solid\n"
        self.facets = []
        # need to add endsolid
    
    def readImage(self, filename):
        img = cv2.imread(filename)
        greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        greyImg = cv2.resize(greyImg, (0, 0), fx=1, fy=1)
        greyImg = cv2.GaussianBlur(greyImg, (9,9), 2.0)
        
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
    
    def magic(self, scale=1, levels=30):
        img = self.image
        n_rows = len(img)
        n_cols = len(img[0])

        # 1) find maximum once
        top = max(max(row) for row in img)

        # 2) draw one big base-plane at height 0.5
        bl = point(0, 0, 0)
        full_x = n_rows * scale
        full_z = n_cols * scale
        self.addRect(
            bl,
            bl + point(0, 0, full_z),
            bl + point(full_x, 0, 0),
            bl + point(full_x, 0, full_z)
        )

        # 3) precompute height for each distinct pixel value
        height_cache = {0: 0.5}
        for val in set(v for row in img for v in row) - {0}:
            layer = (val * levels) // top
            h = layer * (top // levels) / 85 + 0.5
            height_cache[val] = h

        # 4) for each row, run‐length‐encode contiguous equal‐height runs
        for i in tqdm(range(n_rows)):
            # build list of heights for this row
            row_heights = [height_cache[v] for v in img[i]]

            j = 0
            while j < n_cols:
                h = row_heights[j]
                if h == 0.5:
                    # skip the base
                    j += 1
                    continue

                # extend run while same height
                start = j
                while j < n_cols and row_heights[j] == h:
                    j += 1
                length = j - start

                # draw one cuboid spanning `length` pixels in z
                bottom = point(i * scale, 0, start * scale)
                self.addCuboid(
                    bottom,
                    width = scale,           # one pixel in x
                    depth = length * scale,  # run length in z
                    height = h
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
    myLitho = lithograph("calder_test.jpeg")
    # myLitho.addRect(point(0, 0, 0), point(0, 0, 1),  point(1, 0, 0), point(1, 0, 1))
    # myLitho.add3Dpixel(point(0), 7)
    myLitho.magic(1, 10)
    myLitho.saveStlFromArray("calder")
    # myLitho.readImage("jim.png")
    
    
    
if __name__ == "__main__":
    main()