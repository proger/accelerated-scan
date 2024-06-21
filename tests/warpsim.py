# timesteps_per_tile = rows of q
# NUM_WORKERS = number of warps

def load(*args):
    pass

# all to all attention
def sim_full(warpid, n=4, timesteps_per_tile=1, NUM_WORKERS=4):
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
def sim(warpid, NUM_WORKERS, seqlen=4, timestep_tiles_per_thread=1, timesteps_per_tile=1):
    time_stride = timestep_tiles_per_thread*timesteps_per_tile
    qo_blocks = seqlen//(time_stride*NUM_WORKERS)
    
    for q_blk in range(qo_blocks):
        q_seq = (q_blk * NUM_WORKERS + warpid) * time_stride
        q_end = q_seq + time_stride
        load(q_seq)

        #kv_blocks = n//(timesteps_per_tile*NUM_WORKERS)
        #kvbs = [x for x in range(kv_blocks) if x <= q_blk]
        #print(f"{warpid=} {q_blk=} {q_seq=} {kvbs=}")

        for kv_blk in range(q_blk,-1,-1):
            kv_warp_index = kv_blk * NUM_WORKERS + warpid
            kv_seq = kv_warp_index * time_stride # one warp loads the whole kv block to memory
            if q_seq >= kv_seq:
                load(kv_seq)
            #print(f"{warpid=} {q_seq=} {kv_blk=} {kv_seq=}")
            for subtile in range(NUM_WORKERS,-1,-1):
                k_seq = (kv_blk * NUM_WORKERS + subtile) * time_stride # every warp now accesses every block in share memory
                k_end = k_seq + time_stride
                if q_seq >= k_seq:
                    load(k_seq)
                    needs_make_causal = "\\" if q_seq == k_seq else ""
                    print(f"{warpid=} {kv_seq=} q[{q_seq}:{q_end}]@k[{k_seq}:{k_end}] {needs_make_causal}")
        # store here


NUM_WORKERS = 4
for warpid in range(NUM_WORKERS):
    sim(warpid, NUM_WORKERS)
