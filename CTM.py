import cv2
import numpy as np
from random import shuffle
from server.Dsu import DSU



ADAPTIVE_WINSZ = 55  

CCOLORS = [
    (180, 100, 100),
    (180, 130, 100),
    (180, 160, 100),
    (160, 180, 100),
    (130, 180, 100),
    (100, 180, 100),
    (100, 180, 130),
    (100, 180, 160),
    (100, 160, 180),
    (100, 130, 180),
    (100, 100, 180),
    (130, 100, 180),
    (160, 100, 180),
    (180, 100, 160),
    (180, 100, 130),
    (150, 120, 100),
    (100, 150, 120),
    (120, 100, 150),
    (150, 150, 100),
    (100, 150, 150),
    (150, 100, 150),
    (140, 160, 120),
    (120, 140, 160),
    (160, 120, 140),
]

def resize_to_screen(src, maxw=1280, maxh=700, copy=False):

    height, width = src.shape[:2]

    scl_x = float(width)/maxw
    scl_y = float(height)/maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0/scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img

def get_page_extents(small):
    height, width = small.shape[:2]
    xmin = 50   # PAGE_MARGIN_X
    ymin = 20   # PAGE_MARGIN_Y
    xmax = width - 50
    ymax = height - 20

    pagemask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(pagemask, (xmin, ymin), (xmax, ymax), 255, -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])
    return pagemask, outline



def getlines(img):

    moments = cv2.moments(img)

    area = moments['m00']

    cx = moments['m10'] / area
    cy = moments['m01'] / area

    comat = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
        ]) / area
    
    _, svdu, _ = cv2.SVDecomp(comat)

    center = np.array([cx, cy])
    tangent = svdu[:,0].flatten().copy()

    return center, tangent 

def fltp(point):
    return tuple(point.astype(int).flatten())

def proj_x(point, tangent, center):
    return np.dot(tangent, point.flatten()-center)

def euclidean_dist(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def angle_dist(a, b):
    diff = b - a
    
    while(diff > np.pi):
        diff -= 2 * np.pi

    while(diff < -np.pi):
        diff += 2 * np.pi

    return np.abs(diff)

def prelines(img, contour):
    center, tangent = getlines(contour)

    pts = contour.reshape(-1, 2)
    clx = [proj_x(pt, tangent, center) for pt in pts]


    lxmin = min(clx)
    lxmax = max(clx)

    l= center + tangent * lxmin
    r = center + tangent * lxmax

    return center, l, r, tangent


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)



def process(img, mode = 0):

    img = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
      

    if(mode):

        thresh = cv2.adaptiveThreshold(

            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_WINSZ,
            7
        )
        connected = cv2.erode(thresh,    np.ones((1, 3),  np.uint8), iterations=3)
        connected = cv2.dilate(connected, np.ones((2, 8),  np.uint8))
    else:
        thresh = cv2.adaptiveThreshold(

            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_WINSZ,
            25
        )

        connected = cv2.dilate(thresh, box(9,1))
        connected = cv2.erode(connected, box(1,3))


    #cv2.imshow("image2", connected)
    

        # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(connected,4,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to
    # store all the output components
    output = np.zeros((img.shape[0],img.shape[1], 3), dtype="uint8")

    res = []

    pagemask = get_page_extents(img)

    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        h = values[i, cv2.CC_STAT_HEIGHT]
        w = values[i, cv2.CC_STAT_WIDTH]
        x    = values[i, cv2.CC_STAT_LEFT]
        y    = values[i, cv2.CC_STAT_TOP]
        

        if x < 50 or y < 20 or x+w > img.shape[1]-50 or y+h > img.shape[0]-20:
            continue

        if(area < 60):
            continue



        if(mode):
            if(h > 5):
                continue
            if(w / max(h,1) < 10):
                continue
        else:
            if(h / max(w,1) > 0.5):
                continue

            if(w / max(h, 1) > 30):
                continue


            pass

    
        compMask = (label_ids == i).astype("uint8") * 255

        contours, _ = cv2.findContours(compMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        center,l,r, tangent= prelines(compMask, contours[0])

        res.append({"center" : center, "tangent" : tangent, "l" : l, "r" : r, "contour": contours[0], "contoursFull" : contours,  "id" : 0})


        #if(euclidean_dist(fltp(l), fltp(r)) < 150):
            #continue

        #print(fltp(l), fltp(r), euclidean_dist(fltp(l), fltp(r)))

        


        #output = cv2.bitwise_or(output, compMask)


    return output, res

def swap(a, b):
    return b,a

def interval_measure_overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])

def get_local_xrng(contour, center, tangent):

    pts = np.array(contour).reshape(-1, 2)
    projections = [proj_x(pt, tangent, center) for pt in pts]
    return (min(projections), max(projections))

def local_overlap(center, tangent, xrng, b0, b1):

    mn = proj_x(b0, tangent, center)
    mx = proj_x(b1, tangent, center)
    return interval_measure_overlap(xrng, (mn, mx))

def get_weight(a, b, img):

    if(a['l'][0] > b['r'][0]):    
        a,b = swap(a,b)

    overlap = max(local_overlap(a['center'], a['tangent'], get_local_xrng(a['contour'], a['center'], a['tangent']), b['l'], b['r']),
                  local_overlap(b['center'], b['tangent'], get_local_xrng(b['contour'], b['center'], b['tangent']), a['l'], a['r'])
    )

    tangent = b['center'] - a['center']
    angle = np.arctan2(tangent[1], tangent[0])

    angleA = np.arctan2(a['tangent'][1], a['tangent'][0])
    angleB = np.arctan2(b['tangent'][1], b['tangent'][0])

    delta_angle = max(abs(angle_dist(angleA, angle)), abs(angle_dist(angleB, angle))) * 180 / np.pi

    dist = euclidean_dist(a['r'], b['l'])
    vertical_dist = abs(a['center'][1] - b['center'][1])
    horizontal_dist = abs(b['center'][0] - a['center'][0])
    vertical_ratio = vertical_dist / max(horizontal_dist, 1)

    tmp = img.copy()
    #cv2.circle(tmp, fltp(a['center']), 3, (255, 255, 255), 1, cv2.LINE_AA)
    #cv2.circle(tmp, fltp(b['center']), 3, (255, 255, 255), 1, cv2.LINE_AA)
    #cv2.imshow("edge", tmp)
    #cv2.waitKey(0)

    

    #print(vertical_dist)

    if(dist > 150.0 or overlap > 1 or delta_angle > 7.5 or vertical_dist > 5):
        return [1e9, a['id'], b['id']]
    
    return [dist + delta_angle * 10 + vertical_dist*100, a['id'], b['id']]

def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
    return (pts - offset) * scl
    
def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width) * 0.5
    offset = np.array([0.5*width, 0.5*height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval
    

def dfs(node, adj, cnt):

    if(node == -1):
        return

    if(comp[node]):
        return
    comp[node] = cnt
    l = adj[node][0]
    dfs(l, adj, cnt)
    r = adj[node][1]
    dfs(r, adj, cnt)






def get_spans(nodes):
    n = len(nodes)

    #print(nodes)
    cinfo = sorted(nodes, key = lambda x : x['center'][1])

    for i, node in enumerate(cinfo):
        node['id'] = i


    global comp 
    comp = [0 for i in range(n)]
    adj = [[-1,-1] for i in range(n)]

    edges = []
    deg = [0 for i in range(len(nodes))]


    for i in range(1, len(nodes)):
        for j in range(i+1, len(nodes)):

            weight, _, _ = get_weight(cinfo[i], cinfo[j], img)
            if(weight == 1e9):
                continue
            a,b = i,j

            if(cinfo[i]['l'][0] > cinfo[j]['l'][0]):
                a,b = j,i
            edges.append((weight, a , b))



    seen = set()
    deduped = []
    for cost, i, j in edges:
        key = (min(i,j), max(i,j))
        if key not in seen:
            seen.add(key)
            deduped.append((cost, i, j))

    edges = sorted(deduped)

    for edge in edges:
        #print(edge[1], edge[2], n)
        if adj[edge[1]][1] == -1 and adj[edge[2]][0] == -1:

            a = cinfo[edge[1]]
            b = cinfo[edge[2]]
           
            adj[edge[1]][1] = edge[2]
            adj[edge[2]][0] = edge[1]

    cnt = 0


    for i in range(1, n):
        if(comp[i]):
            continue
        cnt+=1
        dfs(i, adj, cnt)

    
    comps = {}

    for i in range(1,n):
        if(comp[i] not in comps):
            comps[comp[i]] = [i]
        else:
            comps[comp[i]].append(i)


    
    res = []

    for i in comps:

        c = comps[i]

        if len(c) > 0:
            x_min = min(cinfo[node]['l'][0] for node in c)
            x_max = max(cinfo[node]['r'][0] for node in c)
            if x_max - x_min > 30:
                res.append(c)
      

    print(res)
    return res, cinfo

def make_tight_mask(contour, xmin, ymin, width, height):
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    cv2.drawContours(tight_mask, [tight_contour], 0, (1, 1, 1), -1)
    return tight_mask

def sample(nodes, spans, shape):

    res = []

    for span in spans:

        cur = []


        for i in span:
            node = nodes[i]
            #print(node['contour'])
            rect = cv2.boundingRect(node["contour"].reshape(-1, 1, 2))
            xmin, ymin, width, height = rect

            mask = make_tight_mask(node['contour'], xmin, ymin, width, height)


            yvals = np.arange(mask.shape[0]).reshape((-1,1))

            tot = (yvals * mask).sum(axis = 0)
            means = tot / mask.sum(axis = 0)
            step = 20

            start = ((len(means)-1) % step)//2

            cur += [(x + xmin, means[x] + ymin) for x in range(start, len(means),step)]
        cur = np.array(cur, dtype=np.float32).reshape((-1, 1, 2))

        cur = pix2norm(shape, cur)

        res.append(cur)
    return res


def visualize_samples(img, spanPoints):
    vis = img.copy()

    if(len(vis.shape) == 2):
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(spanPoints)):
        points = norm2pix(vis.shape, spanPoints[i], as_integer=False)

        mean, eigenvectors = cv2.PCACompute(points.reshape((-1,2)), None, maxComponents=1)

        col = CCOLORS[i % len(CCOLORS)]

        dps = np.dot(points.reshape((-1,2)), eigenvectors.reshape((2,1)))
        dpm = np.dot(mean.flatten(), eigenvectors.flatten())

        l = mean + eigenvectors *(dps.min() - dpm)
        r = mean + eigenvectors * (dps.max() - dpm)

        for p in points:
            cv2.circle(vis, fltp(p), 3, col, -1, cv2.LINE_AA)

        cv2.line(vis, fltp(l), fltp(r), (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("sample points", vis)

def visualize_blobs(img, nodes):

    vis = img.copy()
    for j in range(len(nodes)):
        i = nodes[j]
        cv2.drawContours(vis, i['contoursFull'], 0, CCOLORS[j % len(CCOLORS)], -1)
        cv2.circle(vis, fltp(i['center']), 3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(vis, fltp(i['l']), fltp(i['r']), (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("blobs", vis)


    





#img = cv2.imread("images/IMG_2325.jpg")
#img = cv2.imread("images/IMG_2326.jpg")
#img = cv2.imread("images/IMG_2328.jpg")
img = cv2.imread("images/linguistics_thesis_a.jpg")



cv2.imshow("image1", img)

small = resize_to_screen(img)

tmp = small

img, nodes = process(img)



visualize_blobs(img, nodes)

nodes.insert(0, {"center": np.array([-1e9, -1e9]), "tangent": np.array([0, 0]), 
                  "l": np.array([0, 0]), "r": np.array([0, 0]), 
                  "contour": np.zeros((1, 1, 2), dtype=np.int32),
                  "contoursFull": [], "id": 0})
nodes.reverse()

nodes[1:] = sorted(nodes[1:], key=lambda n: n['center'][1])

idToNodes = [0 for i in range(len(nodes))]

for i in range(1, len(nodes)):
    nodes[i]['id'] = i
    idToNodes[i] = nodes[i]


spans, nodes = get_spans(nodes)

points = sample(nodes, spans, img.shape)


visualize_samples(tmp, points)





#cv2.imshow("image", img)

print("done")
#cv2.imshow("image", img)
cv2.waitKey(0)


