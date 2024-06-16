# timesteps_per_tile = rows of q

NUM_WORKERS = 8 # how many warps

def load(*args):
    pass

# all to all attention
def sim_full(warpid, n=4, timesteps_per_tile=1, NUM_WORKERS=NUM_WORKERS):
    qo_blocks = n//(timesteps_per_tile*NUM_WORKERS)
    kv_blocks = n//(timesteps_per_tile*NUM_WORKERS)
    for q_blk in range(qo_blocks):
        q_seq = (q_blk * NUM_WORKERS + warpid) * timesteps_per_tile
        #print(f"{warpid=} {q_blk=} {q_seq=}")
        for kv_idx in range(kv_blocks):
            kv_seq = (kv_idx * NUM_WORKERS + warpid) * timesteps_per_tile # # one warp loads the whole kv block to memory
            #print(f"{warpid=} {q_seq=} {kv_seq=}")
            for subtile in range(NUM_WORKERS):
                k_index = kv_idx * NUM_WORKERS + subtile # every warp now accesses every block in share memory
                print(f"{warpid=} {kv_seq=} q[{q_seq}]@k[{k_index}]")
        # store here

# causal attention
def sim(warpid, n=8, timesteps_per_tile=1, NUM_WORKERS=NUM_WORKERS):
    qo_blocks = n//(timesteps_per_tile*NUM_WORKERS)
    
    for q_blk in range(qo_blocks):
        q_seq = (q_blk * NUM_WORKERS + warpid) * timesteps_per_tile
        load(q_seq)

        #kv_blocks = n//(timesteps_per_tile*NUM_WORKERS)
        #kvbs = [x for x in range(kv_blocks) if x <= q_blk]
        #print(f"{warpid=} {q_blk=} {q_seq=} {kvbs=}")

        for kv_blk in range(q_blk,-1,-1):
            kv_warp_index = kv_blk * NUM_WORKERS + warpid
            kv_seq = kv_warp_index * timesteps_per_tile # one warp loads the whole kv block to memory
            if q_seq >= kv_seq:
                load(kv_seq)
            #print(f"{warpid=} {q_seq=} {kv_blk=} {kv_seq=}")
            for subtile in range(NUM_WORKERS,-1,-1):
                k_block_index = kv_blk * NUM_WORKERS + subtile # every warp now accesses every block in share memory
                k_index = k_block_index * timesteps_per_tile
                if q_seq >= k_index:
                    load(k_index)
                    print(f"{warpid=} {kv_seq=} q[{q_seq}]@k[{k_index}]")
        # store here


for warpid in range(NUM_WORKERS):
    sim(warpid)
