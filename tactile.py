import cv2

NUM_PIXELS = 150

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
        greyImg = cv2.resize(greyImg, (NUM_PIXELS, NUM_PIXELS))
        return 255 - greyImg
    
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
    
    def addPixelSides(self, bottomLeft, y, invert=False):
        blz = bottomLeft.fixY(0)
        across = blz + point(1, 0, 1)
        self.addRect(blz, blz + point(0, 0, 1), bottomLeft.fixY(y), bottomLeft.fixY(y) + point(0, 0, 1))
        self.addRect(blz, blz + point(1, 0, 0), bottomLeft.fixY(y), bottomLeft.fixY(y) + point(1, 0, 0), invert=True)
        
        self.addRect(across, across - point(0, 0, 1), across.fixY(y), across.fixY(y) - point(0, 0, 1))
        self.addRect(across, across - point(1, 0, 0), across.fixY(y), across.fixY(y) - point(1, 0, 0), invert=True)
        
        
    
    def add2DPixel(self, bottomLeft, y, invert=False):
        self.addRect(bottomLeft.fixY(y), bottomLeft + point(0, y, 1), bottomLeft + point(1, y, 0), bottomLeft + point(1, y, 1), invert)
    
    def add3Dpixel(self, bottomLeft, height):
        self.add2DPixel(bottomLeft.fixY(0), 0, invert=True)
        self.add2DPixel(bottomLeft, height)
        self.addPixelSides(bottomLeft, height)
    
    def magic(self):
        top = max([max(x) for x in self.image])
        heights = list(range(0, top, top // 30))
        # heights.append(top)
        # print(heights)
        for h in heights:
            for i in range(len((self.image))):
                # print(i)
                for j in range(len(self.image[i])):
                    if h == 0:
                        self.add3Dpixel(point(i, 0, j), 0.5)
                        continue
                    if self.image[i][j] >= h:
                        self.add3Dpixel(point(i, 0, j), h / 85 + 0.5) #self.image[i][j] / 85)
        
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
    myLitho = lithograph("mem.png")
    # myLitho.addRect(point(0, 0, 0), point(0, 0, 1),  point(1, 0, 0), point(1, 0, 1))
    # myLitho.add3Dpixel(point(0), 7)
    myLitho.magic()
    myLitho.saveStlFromArray("m&em")
    # myLitho.readImage("jim.png")
    
    
    
if __name__ == "__main__":
    main()