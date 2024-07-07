
#%%

import numpy as np
import matplotlib.pyplot as plt

def tileprint(K, name='K'):
    "format matches tileprint in tk code"
    assert K.shape == (16, 16)
    for laneid in range(32):
        row_top = laneid // 4
        row_bottom = row_top + 8
        col_left = laneid % 4 * 2
        col_right = col_left + 8

        def fmt(r,c,tag):
            odd = "y" in tag
            if odd: # do not print r for odd rows because cuda printf silently runs out of function arguments
                return f"{name}[,{c:02}] {tag}={K[r,c]: .3f}"
            else:
                return f"{name}[{r:02},{c:02}] {tag}={K[r,c]: .3f}"

        print(f"lane={laneid:02}", "    ".join([
            " ".join([fmt(row_top, col_left, "0x"), fmt(row_top, col_left+1, "0y")]),
            " ".join([fmt(row_bottom, col_left, "1x"), fmt(row_bottom, col_left+1, "1y")]),
            " ".join([fmt(row_top, col_right, "2x"), fmt(row_top, col_right+1, "2y")]),
            " ".join([fmt(row_bottom, col_right, "3x"), fmt(row_bottom, col_right+1, "3y")])
        ]))

"""
template <typename H = bf16, int _height, int _width>
__device__ void tileprint(rt<H, _height, _width, ducks::rt_layout::row> reg, char *name) {
    auto laneid = kittens::laneid();
    static_assert(reg.height == 1 && reg.width == 1, "height and width must be 1");
    for(int i = 0; i < reg.height; i++) {
        for(int j = 0; j < reg.width; j++) {
            static_assert(reg.packed_per_thread == 4, "packed_per_thread must be 4");

            int row_top = laneid / 4;
            int row_bottom = row_top + 8;
            int col_left = laneid % 4 * 2; // stride 4
            int col_right = col_left + 8;

            auto item_top_left = __bfloat1622float2(reg.tiles[i][j].data[0]);
            auto item_bottom_left = __bfloat1622float2(reg.tiles[i][j].data[1]);
            auto item_top_right = __bfloat1622float2(reg.tiles[i][j].data[2]);
            auto item_bottom_right = __bfloat1622float2(reg.tiles[i][j].data[3]);
            printf("lane=%02d "
                "%s[%02d,%02d] 0x=% .3f "
                "%s[,%02d] 0y=% .3f    "
                "%s[%02d,%02d] 1x=% .3f "
                "%s[,%02d] 1y=% .3f    "
                "%s[%02d,%02d] 2x=% .3f "
                "%s[,%02d] 2y=% .3f    "
                "%s[%02d,%02d] 3x=% .3f "
                "%s[,%02d] 3y=% .3f\n",
                laneid,
                name, row_top, col_left, item_top_left.x,
                name, col_left+1, item_top_left.y,
                name, row_bottom, col_left, item_bottom_left.x,
                name, col_left+1, item_bottom_left.y,
                name, row_top, col_right, item_top_right.x,
                name, col_right+1, item_top_right.y,
                name, row_bottom, col_right, item_bottom_right.x,
                name, col_right+1, item_bottom_right.y);
        }
    }
}
"""

plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["toolbar"] = "None"
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams['xtick.major.size'] = 0
plt.rcParams['xtick.major.width'] = 0
plt.rcParams['ytick.major.size'] = 0
plt.rcParams['ytick.major.width'] = 0
#plt.rcParams['text.usetex'] = True

dotdata = np.arange(8)//2
xy = ['x', 'y'] * 4
xticklabels = [f'{d}{x}' for d, x in zip(dotdata, xy)]

@np.vectorize
def coord(laneid, bottom, right):
    row_top = laneid // 4
    row_bottom = row_top + 8
    colL = laneid % 4 * 2 # stride 4
    colR = colL + 8
    row = row_bottom if bottom else row_top
    col = colR if right else colL
    return row, col

@np.vectorize
def tdcoord_to_laneid(row, col):
    return (row % 8) * 4 + (col % 8) // 2

lanes = np.arange(32)[:, None, None]
top = np.arange(2)[None, None, :] # 0 for top or 1 for bottom
left = np.arange(2)[None, :, None] # 0 for left or 1 for right

coords = coord(lanes, top, left)
rows, cols = coords
rows = rows.reshape(32, -1)
cols = cols.reshape(32, -1)

rows_xy = np.repeat(rows, 2, axis=1)
cols_xy = np.repeat(cols, 2, axis=1)

#fig, ax = plt.subplots(figsize=(8, 8))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('rt<bf16, 1, 1, row> layout')

ax.matshow(rows_xy, aspect='auto', cmap='coolwarm', vmin=-2, vmax=18)
ax.set_yticks(np.arange(32))
ax.set_title(f'Lane and Register to [Row, Column]')
ax.set_ylabel('thread lane')
ax.set_xlabel('thread registers, $0x$ stands for .data[0].x')
ax.set_xticks(np.arange(8))
# remove spine
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticklabels([f'${xt}$' for xt in xticklabels])

x, y = np.meshgrid(np.arange(16), np.arange(16))
@np.vectorize
def cat(x, y):
    return f'{x},{y}'
rowcol_to_loc = cat(x,y)
rowcol_to_r = np.zeros((16,16))

for (i, j), r in np.ndenumerate(rows_xy):
    c =  cols_xy[i,j] + j%2
    laneid = tdcoord_to_laneid(r, c)
    rowcol_to_loc[r,c] = f'{laneid}:{xticklabels[j]}'
    rowcol_to_r[r,c] = r
    ax.text(j, i, f'[{str(r).rjust(2)},{str(c).rjust(2)}]', ha='center', va='center')

# highlight the diagonal
for i in range(16):
    rowcol_to_r[i,i] = 8.5

bx.matshow(rowcol_to_r, aspect='auto', cmap='coolwarm', vmin=-2, vmax=18)
bx.set_xticks(np.arange(16))
bx.set_yticks(np.arange(16))
# remove spine
bx.spines['top'].set_visible(False)
bx.spines['right'].set_visible(False)
bx.spines['bottom'].set_visible(False)
bx.spines['left'].set_visible(False)
bx.set_title('Tile Location to lane:register, $0x$ stands for .data[0].x')
bx.set_xlabel('tile columns')
bx.set_ylabel('tile rows')

for (i, j), r in np.ndenumerate(rowcol_to_loc):
    bx.text(j, i, r, ha='center', va='center')

plt.savefig('tile_layout.pdf', dpi=300, bbox_inches='tight')
plt.savefig('tile_layout.png', dpi=300, bbox_inches='tight')
# %%
