from multiprocessing import Pool
import os

import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def create_headers():
    headers = {}
    headers['cycle'] = [0]
    headers['ball'] = [1, 2, 3, 4, 5, 6, 7, 8]

    # 4 5 6, 7 8 9
    for i in range(1, 12):
        headers[f'tm-{i}-noise'] = list(range(9 + (i - 1) * 8, 9 + i * 8))  # max=4+11*3 = 37
        headers[f'opp-{i}-noise'] = list(range(97 + (i - 1) * 8, 97 + i * 8))  # max = 37+11*3 = 33+37 = 70
        headers[f'tm-{i}-full'] = list(range(8 + 185 + (i - 1) * 8, 8 + 185 + i * 8))  # max=70 + 33 = 103
        headers[f'opp-{i}-full'] = list(range(8 + 273 + (i - 1) * 8, 8 + 273 + i * 8))
        
    return headers

def contain(x_low, y_low, x_high, y_high, p):
    x, y = p[0], p[1]
    if x_low < x < x_high and y_low < y < y_high:
        return True
    return False

def pos_list(args):
    data = np.genfromtxt(f'{folder}{file}', delimiter=',')[:, :]
    headers = args[0]
    n = args[1]
    print(f'{n}/100')
    
    
    episodes = []
    last_cycle = None
    episode_start = None
    index_start = None
    for i in range(data.shape[0]):
        cycle = data[i][0]
        if last_cycle is None:
            last_cycle = cycle
            episode_start = cycle
            index_start = i
            continue

        if cycle - 1 != last_cycle:
            episodes.append((index_start, i - 1, episode_start, last_cycle))
            episode_start = cycle
            index_start = i
        last_cycle = cycle
    

    
    poses = [[[] for _ in range(10)] for _ in range(11)]
    for u in range(1, 12):
        for N in range(10):
            player_poses = []
            for ep in episodes:
                if ep[3] - ep[2] < N:
                    continue
                for j in range(ep[0], ep[1] + 1 - N):
                    seen_pos = data[j, headers[f'opp-{u}-full'][:2]]
                    predict_pos = data[j+N, headers[f'opp-{u}-full'][:2]]
                    
                    rpos = predict_pos - seen_pos
                    player_poses.append(rpos)
        
            poses[u-1][N] += player_poses
    return poses

def create_pickle():
    folder = '/data1/aref/2d/data/9/'
    files = os.listdir(folder)
    headers = create_headers()

    inps = []
    for n, file in enumerate(files):
        print(file)
        if not file.endswith('.csv'):
            continue
        inp = (headers, n)
        inps.append(inp)

    pool = Pool(50)
    res = pool.map(pos_list, inps)

    poses = [[[] for _ in range(10)] for _ in range(11)]
    for i, r in enumerate(res):
        print(i)
        for u in range(11):
            for N in range(10):
                poses[u][N] += r[u][N]
                
                    
    pk.dump(poses, open('prob_pos.pkl', 'wb'))

def read_pickle():
    data = pk.load(open('prob_pos.pkl', 'rb'))

    s = ''
    m = 1_000_000
    for i in range(len(data)):
        for j in range(len(data[i])):
            s += f'{len(data[i][j])}\t'
            if m > len(data[i][j]):
                m = len(data[i][j])
        s += '\n'
    print(s)        
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j][:m]
    data = np.array(data)
    np.save('data.npy', data)

def draw():
    data = np.load('data.npy')
    print(data.shape)

    plt.scatter(data[6, 1, :, 0], data[6, 1, :, 1],s=1)
    plt.savefig('test1',dpi=500)
    

def prob_draw():
    data = np.load('data.npy')
    pos = data[6, 1, :]
    min_x = np.min(pos[:, 0])
    max_x = np.max(pos[:, 0])
    min_y = np.min(pos[:, 1])
    max_y = np.max(pos[:, 1])

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    size = min(delta_x, delta_y)/10
    step = size/10

    xs = np.arange(min_x, max_x-size,step)
    ys = np.arange(min_y, max_y-size,step)
    prob = np.zeros((xs.shape[0], ys.shape[0]))
    for i, x in enumerate(xs):
        print(f'{i}/{len(xs)}')
        for j, y in enumerate(ys):
            c = \
                (pos[:, 0] < x + size) \
                * (x < pos[:, 0]) \
                * (pos[:, 1] < y + size) \
                * (y < pos[:, 1]) \
                    
            prob[i][j] += np.sum(c)
                    
    prob /= len(pos)
        
    # X = xs
    # Y = ys
    # X, Y = np.meshgrid(X, Y)
            
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, prob.T)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('PROB')

    plt.imshow(prob)
    plt.savefig('prob-heat')

data = np.load('data.npy')
print(data.shape)
# exit()


# plt.plot(pos[:,0], pos[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(pos[simplex, 0], pos[simplex, 1], 'k-')
    
# plt.plot(pos[hull.vertices,0], pos[hull.vertices,1], 'r--', lw=2)
# plt.plot(pos[hull.vertices[0],0], pos[hull.vertices[0],1], 'ro')
# print(hull.vertices)
# plt.savefig('convex')

for u in range(data.shape[0]):
    print(f'u={u}')
    f = open(f'vertices/v-{u}', 'w')
    for c in range(1,data.shape[1]):
        pos = data[u, c, :]
        hull = ConvexHull(pos)
        
        f.write(f'# {c} {len(hull.vertices)}\n')
        for v in hull.vertices:
            p = pos[v, 0], pos[v, 1]
            f.write(f'{p[0]} {p[1]}\n')
            
        
    f.close()



            