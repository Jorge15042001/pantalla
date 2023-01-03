import numpy as np
import cv2
import pandas as pd


def get_intersect(a1, a2, b1, b2):
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return np.array([x/z, y/z])


def norm(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5


class Points():
    def __init__(self):
        self.points = [[100, 100], [100, 200]]
        self.selected = None

    def getPointAt(self, x, y):
        for p in self.points:
            if norm(x, y, p[0], p[1]) < 10:
                self.selected = p
                return p
        self.selected = None

    def unselect(self):
        self.selected = None


im_points = Points()

x_list = []
y_list = []
z_list = []
lines = []

#  plt.show(block=False)


df = pd.read_csv('coordinates.csv')
data = np.array(df).astype(int)
P1 = data[0]
P2 = data[1]
P3 = data[2]
P4 = data[3]
P5 = data[4]
P6 = data[5]
P7 = data[6]


X1_e1 = P1
X1_e2 = P2

X2_e1 = P3
X2_e2 = P4

X3_e1 = P5
X3_e2 = P6

Y1_e1 = P1
Y1_e2 = P7

Y2_e1 = P3
Y2_e2 = P5

Y3_e1 = P4
Y3_e2 = P6

Z1_e1 = P1
Z1_e2 = P3

Z2_e1 = P2
Z2_e2 = P4

Z3_e1 = P7
Z3_e2 = P5

wo = P1

ref_x = P2
ref_y = P7
ref_z = P3

ref_x = np.array([ref_x])
ref_y = np.array([ref_y])
ref_z = np.array([ref_z])

ax1, bx1, cx1 = np.cross(X1_e1, X1_e2)
ax2, bx2, cx2 = np.cross(X2_e1, X2_e2)
Vx = np.cross([ax1, bx1, cx1], [ax2, bx2, cx2])
Vx = Vx/Vx[2]

ay1, by1, cy1 = np.cross(Y1_e1, Y1_e2)
ay2, by2, cy2 = np.cross(Y2_e1, Y2_e2)
Vy = np.cross([ay1, by1, cy1], [ay2, by2, cy2])
Vy = Vy/Vy[2]

az1, bz1, cz1 = np.cross(Z1_e1, Z1_e2)
az2, bz2, cz2 = np.cross(Z2_e1, Z2_e2)
Vz = np.cross([az1, bz1, cz1], [az2, bz2, cz2])
Vz = Vz/Vz[2]

length_x = np.sqrt(np.sum(np.square(ref_x - wo)))
length_y = np.sqrt(np.sum(np.square(ref_y - wo)))
length_z = np.sqrt(np.sum(np.square(ref_z - wo)))

ref_x = np.array(ref_x)
ref_y = np.array(ref_y)
ref_z = np.array(ref_z)
wo = np.array(wo)
Vx = np.array(Vx)
Vy = np.array(Vy)
Vz = np.array(Vz)


ax, resid, rank, s = np.linalg.lstsq((Vx-ref_x).T, (ref_x - wo).T)
ax = ax[0][0]/length_x

ay, resid, rank, s = np.linalg.lstsq((Vy-ref_y).T, (ref_y - wo).T)
ay = ay[0][0]/length_y

az, resid, rank, s = np.linalg.lstsq((Vz-ref_z).T, (ref_z - wo).T)
az = az[0][0]/length_y

px = ax*Vx
py = ay*Vy
pz = az*Vz

P = np.empty([3, 4])
P[:, 0] = px
P[:, 1] = py
P[:, 2] = pz
P[:, 3] = wo

Hxy = np.zeros((3, 3))
Hyz = np.zeros((3, 3))
Hzx = np.zeros((3, 3))

Hxy[:, 0] = px
Hxy[:, 1] = py
Hxy[:, 2] = wo

Hyz[:, 0] = py
Hyz[:, 1] = pz
Hyz[:, 2] = wo

Hzx[:, 0] = px
Hzx[:, 1] = pz
Hzx[:, 2] = wo

Hxy_inv = np.linalg.inv(Hxy)
Hyz_inv = np.linalg.inv(Hyz)
Hzx_inv = np.linalg.inv(Hzx)

#  PlaneTransform = cv2.getPerspectiveTransform(np.array([P1,P7,P2]),np.array([P3,P4,P5]))


def drawPointsEyeFeet(img, im_points, ratio, padding):
    p_obj_high = np.array(im_points[0])
    p_obj_low = np.array(im_points[1])

    cv2.line(img, p_obj_high, p_obj_low, (100, 100, 100), 1)  # blue
    pointAtHorizon = get_intersect(Vx[:2], Vy[:2], p_obj_low, P1[:2])
    inter = get_intersect(P3[:2], pointAtHorizon, p_obj_high, p_obj_low)
    rel_h = (p_obj_high-p_obj_low)/(inter-p_obj_low)
    print(rel_h)

    cv2.line(img, P1[:2], pointAtHorizon.astype(
        int), (255, 255, 255), 1)  # blue
    cv2.line(img, im_points.points[0], pointAtHorizon.astype(
        int), (255, 255, 255), 1)  # blue
    cv2.line(img, P3[:2], pointAtHorizon.astype(
        int), (255, 255, 255), 1)  # blue


def drawPoints(img, im_points, ratio=1, padding=(0, 0)):
    p_obj_high = np.array(im_points[0]).astype(int)
    p_obj_low = np.array(im_points[1]).astype(int)

    vy_corrected = ((Vy[:-1] * ratio)+padding).astype(int)
    vx_corrected = ((Vx[:-1] * ratio)+padding).astype(int)
    p1_corrected = ((P1[:-1] * ratio)+padding).astype(int)
    p3_corrected = ((P3[:-1] * ratio)+padding).astype(int)

    cv2.line(img, p_obj_high, p_obj_low, (100, 0, 0), 30)  # blue
    pointAtHorizon = get_intersect(
        vx_corrected[:2], vy_corrected[:2], p_obj_low, p1_corrected[:2])
    inter = get_intersect(
        p3_corrected[:2], pointAtHorizon, p_obj_high, p_obj_low)
    rel_h = (p_obj_high-p_obj_low)/(inter-p_obj_low)
    print(rel_h)

    cv2.line(img, p1_corrected[:2], pointAtHorizon.astype(
        int), (255, 255, 255), 1)  # blue
    cv2.line(img, p_obj_high, pointAtHorizon.astype(
        int), (255, 255, 255), 1)  # blue
    cv2.line(img, p3_corrected[:2], pointAtHorizon.astype(
        int), (255, 255, 255), 1)  # blue


def drawSurface(img, ratio=1, padding=(0, 0)):
    #  padding = padding [::-1]
    #  padding = (padding[0] * 2, padding[1] * 2)
    po1 = (Vy[:-1] * ratio)+padding
    po2 = (Vx[:-1] * ratio)+padding

    cv2.line(img, po1.astype(int),
             po2.astype(int), (255, 255, 0), 1)  # blue
    for i in range(10):
        p1 = np.array([[[i*50, 0]]], dtype=np.float32)
        p2 = np.array([[[0, i*50]]], dtype=np.float32)
        pf1 = (cv2.perspectiveTransform(p1, Hxy)[0][0]*ratio)+padding
        pf2 = (cv2.perspectiveTransform(p2, Hxy)[0][0]*ratio)+padding
        cv2.line(img, pf1.astype(int), po1.astype(int), (0, 255, 0), 1)  # blue
        cv2.line(img, pf2.astype(int), po2.astype(int), (0, 255, 0), 1)  # blue


def drawReferenceHeigth(img):

    cv2.line(img, wo[:-1], P3[:-1], (0, 0, 255), 2)  # blue
    cv2.line(img, Vy[:-1].astype(int), P3[:-1], (0, 0, 255), 2)  # blue
    cv2.line(img, Vx[:-1].astype(int), P3[:-1], (0, 0, 255), 2)  # blue

    for line in lines:
        cv2.line(img, line[0], line[1], (255, 0, 0), 1)


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        p = np.array([[[x, y]]], dtype=np.float32)  # imgae coordinates
        pf1 = cv2.perspectiveTransform(p, Hxy_inv)[0][0]  # real coordinates
        x1, y1 = pf1  # 3D space coordinates up to scale

    if event == cv2.EVENT_LBUTTONDOWN:
        point = im_points.getPointAt(x, y)
        if point is None:
            return
    if event == cv2.EVENT_MOUSEMOVE and im_points.selected is not None:
        im_points.selected[0] = x
        im_points.selected[1] = y
    if event == cv2.EVENT_LBUTTONUP:
        im_points.unselect()


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    while True:

        #  img = cv2.imread('image2.jpg')
        flag_frame, img = vid.read()
        r, c, temp = img.shape

        drawSurface(img)
        #  drawReferenceHeigth(img)
        drawPoints(img, im_points)

        cv2.imshow("img", img)
        cv2.setMouseCallback('img', click_event)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
