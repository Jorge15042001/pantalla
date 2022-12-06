import cv2
import numpy as np
import pandas as pd


def norm(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

class Points():
    def __init__(self):
        df = pd.read_csv('./coordinates.csv');
        data = np.array(df).astype(int)
        data= list(data)
        data = list(map(lambda x:list(x), data))
        self.points = data
        self.selected = None
    def getPointAt(self,x,y):
        for p in self.points:
            if norm(x, y, p[0],p[1])<10 :
                self.selected = p
                return p
        self.selected = None




def click_event(event, x, y, flags, params):
    if event==cv2.EVENT_LBUTTONDOWN:
        point  = im_points.getPointAt( x, y)
        if point == None : return
        print("cliked -> ",x,y, "selected", point)
    if event == cv2.EVENT_MOUSEMOVE and im_points.selected!=None:
        print("moving")
        im_points.selected[0]=x
        im_points.selected[1]=y
    if event==cv2.EVENT_LBUTTONUP:
        print("...leaving")
        im_points.selected = None
    
def drawPoints(img,im_points):
     
    P1 = im_points.points[0]# front down
    P2 = im_points.points[1]# down  rigth
    P3 = im_points.points[2]# front up
    P4 = im_points.points[3]# up    rigth
    P5 = im_points.points[4]# up    left
    P6 = im_points.points[5]# up           back back
    P7 = im_points.points[6]# down  left

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



###  X  ###
    cv2.line(img,(X1_e1[0],X1_e1[1]),(X1_e2[0],X1_e2[1]),(0,0,255),1)        #blue
    cv2.line(img,(X2_e1[0],X2_e1[1]),(X2_e2[0],X2_e2[1]),(0,0,255),1)
    cv2.line(img,(X3_e1[0],X3_e1[1]),(X3_e2[0],X3_e2[1]),(0,0,255),1)

###  Y  ###
    cv2.line(img,(Y1_e1[0],Y1_e1[1]),(Y1_e2[0],Y1_e2[1]),(0,255,0),1)     #green
    cv2.line(img,(Y2_e1[0],Y2_e1[1]),(Y2_e2[0],Y2_e2[1]),(0,255,0),1)
    cv2.line(img,(Y3_e1[0],Y3_e1[1]),(Y3_e2[0],Y3_e2[1]),(0,255,0),1)

###  Z  ###
    cv2.line(img,(Z1_e1[0],Z1_e1[1]),(Z1_e2[0],Z1_e2[1]),(250,0,0),1)      #red
    cv2.line(img,(Z2_e1[0],Z2_e1[1]),(Z2_e2[0],Z2_e2[1]),(250,0,0),1)
    cv2.line(img,(Z3_e1[0],Z3_e1[1]),(Z3_e2[0],Z3_e2[1]),(255,0,0),1)

def projections(img,im_points):
    P1 = im_points.points[0]
    P2 = im_points.points[1]
    P3 = im_points.points[2]
    P4 = im_points.points[3]
    P5 = im_points.points[4]
    P6 = im_points.points[5]
    P7 = im_points.points[6]


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


    try:
        ax1,bx1,cx1 = np.cross(X1_e1,X1_e2)
        ax2,bx2,cx2 = np.cross(X2_e1,X2_e2)
        Vx = np.cross([ax1,bx1,cx1],[ax2,bx2,cx2])
        Vx = Vx/Vx[2]

        ay1,by1,cy1 = np.cross(Y1_e1,Y1_e2)
        ay2,by2,cy2 = np.cross(Y2_e1,Y2_e2)
        Vy = np.cross([ay1,by1,cy1],[ay2,by2,cy2])
        Vy = Vy/Vy[2]

        az1,bz1,cz1 = np.cross(Z1_e1,Z1_e2)
        az2,bz2,cz2 = np.cross(Z2_e1,Z2_e2)
        Vz = np.cross([az1,bz1,cz1],[az2,bz2,cz2])
        Vz = Vz/Vz[2]

        length_x = np.sqrt(np.sum(np.square(ref_x - wo)))   
        length_y = np.sqrt(np.sum(np.square(ref_y - wo)))   
        length_z = np.sqrt(np.sum(np.square(ref_z - wo)))   

        print("lengths: ", length_x,length_y,length_z)


        ref_x = np.array(ref_x)
        ref_y = np.array(ref_y)
        ref_z = np.array(ref_z)
        wo = np.array(wo)
        Vx = np.array(Vx)
        Vy = np.array(Vy)
        Vz = np.array(Vz)


        ax,resid,rank,s = np.linalg.lstsq( (Vx-ref_x).T , (ref_x - wo).T )
        ax = ax[0][0]/length_x

        ay,resid,rank,s = np.linalg.lstsq( (Vy-ref_y).T , (ref_y - wo).T )
        ay = ay[0][0]/length_y

        az,resid,rank,s = np.linalg.lstsq( (Vz-ref_z).T , (ref_z - wo).T )
        az = az[0][0]/length_y

        px = ax*Vx
        py = ay*Vy
        pz = az*Vz

        P = np.empty([3,4])
        P[:,0] = px
        P[:,1] = py
        P[:,2] = pz
        P[:,3] = wo

        Hxy = np.zeros((3,3))
        Hyz = np.zeros((3,3))
        Hzx = np.zeros((3,3))

        Hxy[:,0] = px
        Hxy[:,1] = py
        Hxy[:,2] = wo

        Hyz[:,0] = py
        Hyz[:,1] = pz
        Hyz[:,2] = wo

        Hzx[:,0] = px
        Hzx[:,1] = pz
        Hzx[:,2] = wo


        #  Hxy[0,2] = Hxy[0,2]
        #  Hxy[1,2] = Hxy[1,2]
        #
        #  Hyz[0,2] = Hyz[0,2] + 100
        #  Hyz[1,2] = Hyz[1,2] + 100
        #
        #  Hzx[0,2] = Hzx[0,2] - 50
        #  Hzx[1,2] = Hzx[1,2] + 50


        r,c,temp = img.shape

        Txy = cv2.warpPerspective(img,Hxy,(r,c),flags=cv2.WARP_INVERSE_MAP)
        Tyz = cv2.warpPerspective(img,Hyz,(r,c),flags=cv2.WARP_INVERSE_MAP)
        Tzx = cv2.warpPerspective(img,Hzx,(r,c),flags=cv2.WARP_INVERSE_MAP)

        return Txy,Tyz,Tzx
    except:
        return img,img,img
        print("can't compute")

im_points = Points()
vid = cv2.VideoCapture(0)

while True:
    flag_frame , img = vid.read()
    if not flag_frame: continue

    drawPoints(img, im_points)
    Txy,Tyz,Tzx = projections(img, im_points)

    Txy= cv2.putText(Txy, 'Txy', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 6, (255,0,0), 2, cv2.LINE_AA)
    TyZ= cv2.putText(Tyz, 'Tyz', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 6, (255,0,0), 2, cv2.LINE_AA)
    Tzx= cv2.putText(Tzx, 'Tzx', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 6, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("img",img)
    cv2.imshow("Txy",Txy)
    cv2.imshow("Tyz",Tyz)
    cv2.imshow("Tzx",Tzx)

    cv2.setMouseCallback('img', click_event)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

f = open("./coordinates.csv","w")
f.write("X,Y,Z\n")
for p in im_points.points:
    f.write("%d,%d,%d\n"%(p[0],p[1],1))
f.close()
