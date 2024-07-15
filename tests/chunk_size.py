#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams['font.family'] = 'serif'


dim = np.array([16,32,64,128,256,512,1024])
chunk_size = np.array([8,16,32,64,128,256,512])

colormap = plt.get_cmap('viridis')
num_colors = len(dim)
colors = [colormap(i) for i in np.linspace(0, 1, num_colors)]


plt.figure(figsize=(10,7))

words = chunk_size[:,None] * dim
state = dim*dim

# plot horizontal lines for state
for i in range(len(dim)):
    plt.axhline(y=state[i], color=colors[i], linestyle=':', label=f'${dim[i]}^2$', linewidth=1)
    d = dim[i]

    # plt.plot(chunk_size, 2 * chunk_size * d, color=colors[i], label=f'$2C\cdot {dim[i]}$', linewidth=1)
    # c = d / 2
    # plt.scatter(c, 2*c*d, color=colors[i], label=f'breakeven chunk for dim {dim[i]}', s=50)

    plt.plot(chunk_size, 3 * chunk_size * d, color=colors[i], label=f'$3C\cdot {dim[i]}$', linewidth=1)
    c = d / 3
    plt.scatter(c, 3*c*d, color=colors[i], label=f'saturation length for dim {dim[i]} is {int(c)}', s=50)

plt.title('Until when does chunking reduce memory usage?')

# powers of 2 y axis
plt.yscale('log', base=2)
# integers x axis
plt.xticks(chunk_size)
plt.xlabel('chunk size')
plt.ylabel('used memory words')

plt.legend()

plt.savefig('chunk_size.pdf', bbox_inches='tight', dpi=300)