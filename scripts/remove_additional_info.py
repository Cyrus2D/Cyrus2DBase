import os

files = os.listdir('data/')
for file in files:
    if file.split('.')[-1] != 'csv':
        continue
    
    print(f'data/{file}')
    f = open(f'data/{file}', 'r')
    lines = f.read().split('\n')
    print(lines[0])
    lines[0] = ','.join(lines[0].split(',')[:-4])
    f.close()
    f = open(f'data/{file}', 'w')
    f.write('\n'.join(lines))
