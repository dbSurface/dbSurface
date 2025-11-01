def check_and_update_usage(
    num_shards: int,
    run_dir: str,
    remaining_rows: int,
    database_id: str,
    total_rows_estimate_int: int,
    supabase_client,
    vol,
) -> None:
    import pyarrow.dataset as ds
    import os, glob
    import glob
    import time

    MAX_WAIT = 60
    step = 2
    waited = 0

    while (
        True
    ):  # we wait to make sure all shards' updates have been committed--can't commit in each shard due to volume concurrent writer limit
        vol.reload()
        arrow_files = glob.glob(os.path.join(run_dir, "*.arrow"))
        if len(arrow_files) == num_shards:
            break
        if waited >= MAX_WAIT:
            raise RuntimeError(
                f"Only {len(arrow_files)}/{num_shards} shards committed "
                "after a full minute – giving up."
            )
        time.sleep(step)
        waited += step

    ds_obj = ds.dataset(run_dir, format="arrow")
    num_rows = ds_obj.count_rows()

    db_res = (
        supabase_client.table("databases")
        .select("user_id")
        .eq("database_id", database_id)
        .single()
        .execute()
    )
    user_id = db_res.data["user_id"]
    user_res = (
        supabase_client.table("users")
        .select("monthly_projected_rows")
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    current_monthly_rows = user_res.data["monthly_projected_rows"]
    difference = num_rows - total_rows_estimate_int
    monthly_rows = current_monthly_rows + difference
    supabase_client.table("users").update({"monthly_projected_rows": monthly_rows}).eq(
        "user_id", user_id
    ).execute()

    if num_rows > remaining_rows:
        raise RuntimeError(f"Usage limit exceeded: cancelling reduce.")


def get_num_shards(total_rows: int, creds) -> int:
    import math

    if total_rows < 100_000:
        rows_per_shard = 5_000
    elif total_rows < 400_000:
        rows_per_shard = 10_000
    elif total_rows < 1_000_000:
        rows_per_shard = 15_000
    elif total_rows < 5_000_000:
        rows_per_shard = 35_000
    else:
        rows_per_shard = 60_000

    hard_cap = 80

    free_slots = _free_connection_slots(creds, reserve=5)

    return min(
        math.ceil(total_rows / rows_per_shard),
        hard_cap,
        free_slots,
    )


def _free_connection_slots(creds, reserve=5):
    import psycopg

    with psycopg.connect(**creds, prepare_threshold=None) as pg:
        with pg.cursor() as cur:
            cur.execute("SHOW max_connections")
            max_conn = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM pg_stat_activity")
            in_use = int(cur.fetchone()[0])
    print("max available connections and in use", max_conn, in_use)

    return max(1, max_conn - in_use - reserve)


def get_creds(database_id: str):
    from supabase import create_client
    import os

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    db = (
        sb.table("databases")
        .select("db_name,db_host,db_port,restricted_db_user,restricted_db_password")
        .eq("database_id", database_id)
        .execute()
    ).data[
        0
    ]  # will raise if nothing
    creds = dict(
        dbname=db["db_name"],
        user=db["restricted_db_user"],
        password=db["restricted_db_password"],
        host=db["db_host"],
        port=db["db_port"],
    )

    return creds


def _flush(buf, col_names, writer, sink):
    import pyarrow as pa, pyarrow.ipc as ipc
    import pyarrow.compute as pc
    import logging

    rows = [dict(zip(col_names, vs)) for vs in zip(*buf.values())]
    batch = pa.RecordBatch.from_pylist(rows)

    if writer is None:
        writer = ipc.new_file(sink, batch.schema)

        logging.info("flushing batch with %s rows", batch.num_rows)

    # -------- null-column check ---------- #
    for i, name in enumerate(batch.schema.names):
        arr = batch.column(i)  # RecordBatch → column by index
        if pc.all(pc.is_null(arr)).as_py():
            logging.warning("⚠︎ column %s became 100%% null in this batch", name)
    writer.write_batch(batch)

    for k in buf:
        buf[k].clear()

    return writer


def fetch_table_helper(
    shard_id: int,
    schema: str,
    table: str,
    vector_col: str,
    primary_key_col: str,
    total_shards: int,
    creds,
    run_dir: str,
):
    import psycopg, pyarrow as pa, pyarrow.ipc as ipc, pathlib
    from pgvector.psycopg import register_vector
    import uuid
    import time

    pg = psycopg.connect(**creds)
    with pg.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
    register_vector(pg)
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, udt_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position
        """,
            (schema, table),
        )
        rows = cur.fetchall()
        col_names, udt_names = zip(*rows)

    dst = pathlib.Path(run_dir) / f"{shard_id}.arrow"
    tmp = dst.with_suffix(".tmp")
    sink = pa.OSFile(str(tmp), "wb")
    writer = None
    buf = {c: [] for c in col_names}

    copy_sql = f"""
    COPY (
        SELECT *
            FROM {schema}.{table}
            WHERE mod(abs(hashtext(({primary_key_col})::text)), %s) = %s
    ) TO STDOUT (FORMAT BINARY)
"""
    with pg.cursor() as cur, cur.copy(copy_sql, (total_shards, shard_id)) as cp:
        cp.set_types(list(udt_names))
        MAX_ROWS = 1000
        MAX_BYTES = 16 * 1024**2  # 16mb

        batch_rows = 0
        batch_bytes = 0
        t_read = 0.0  # time waiting inside cp.read_row()
        t_parse = 0.0  # loop bookkeeping / conversions
        t_flush = 0.0  # Arrow write + file I/O
        t0_batch = time.perf_counter()

        while True:
            t0 = time.perf_counter()
            row = cp.read_row()
            t_read += time.perf_counter() - t0

            if row is None:  # end-of-stream
                break

            t0 = time.perf_counter()
            row_size_est = 0
            for name, val in zip(col_names, row):
                if name == vector_col:
                    row_size_est += val.nbytes
                elif isinstance(val, str):
                    row_size_est += len(val)
                else:
                    row_size_est += 8
                if isinstance(val, uuid.UUID):
                    val = str(val)
                buf[name].append(val)
            t_parse += time.perf_counter() - t0

            batch_rows += 1
            batch_bytes += row_size_est

            if (batch_rows >= MAX_ROWS) or (batch_bytes >= MAX_BYTES):
                t0 = time.perf_counter()
                writer = _flush(buf, col_names, writer, sink)
                t_flush += time.perf_counter() - t0
                dt = time.perf_counter() - t0_batch
                # print( #prints each batch of 1000
                #     f"⤵︎ batch {batch_rows:5} rows | "
                #     f"{batch_bytes/1_048_576:5.1f} MiB | "
                #     f"{batch_rows/dt:7.0f} r/s | "
                #     f"{batch_bytes*8/dt/1_000_000:6.1f} Mb/s | "
                #     f"read {t_read:.2f}s, parse {t_parse:.2f}s, "
                #     f"flush {t_flush:.2f}s",
                #     f"{shard_id} shard_id",
                # )
                # reset counters
                batch_rows = batch_bytes = 0
                t_read = t_parse = t_flush = 0.0
                t0_batch = time.perf_counter()

        if batch_rows:  # final tail
            t1 = time.perf_counter()
            writer = _flush(buf, col_names, writer, sink)
            t_flush += time.perf_counter() - t1
            dt = time.perf_counter() - t0_batch

    pg.close()

    if writer is not None:
        writer.close()
    sink.close()
    tmp.rename(dst)


def embed_to_2d_helper(
    vector_col: str,
    run_dir: str,
    num_shards: int,
    vol,
    target_gpu_mem: float = 7e9,  # T4 can handle 16GB, so leaving headroom
    n_neighbors: int = 15,
    batch_rows: int = 100_000,
):
    """
    Write primary_key, x, y to /cache/xy.arrow
    """
    import pyarrow.dataset as ds, pyarrow as pa, pyarrow.ipc as ipc
    import numpy as np, cupy as cp, math, os, tempfile, gc, glob
    from cuml.manifold import UMAP
    import glob
    import multiprocessing as mp
    import time

    MAX_WAIT = 60
    step = 2
    waited = 0

    while (
        True
    ):  # we wait to make sure all shards' updates have been committed--can't commit in each shard due to volume concurrent writer limit
        vol.reload()
        arrow_files = glob.glob(os.path.join(run_dir, "*.arrow"))
        if len(arrow_files) == num_shards:
            break
        if waited >= MAX_WAIT:
            raise RuntimeError(
                f"Only {len(arrow_files)}/{num_shards} shards committed "
                "after a full minute – giving up."
            )
        time.sleep(step)
        waited += step

    ds_obj = ds.dataset(run_dir, format="arrow")
    num_rows = ds_obj.count_rows()
    print("num rows: ", num_rows)

    ds_obj = ds.dataset(run_dir, format="arrow")
    dim = len(ds_obj.take([0])[vector_col][0])
    print("vector dim =", dim)

    tmp = tempfile.NamedTemporaryFile(prefix="vectors_", suffix=".mmp", delete=False)
    mmap_path = tmp.name
    tmp.close()

    mmap = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(num_rows, dim))

    out = 0
    for batch in ds_obj.scanner(
        columns=[vector_col], batch_size=batch_rows
    ).to_batches():
        vecs = batch.column(vector_col).values.to_numpy().reshape(-1, dim)
        n = vecs.shape[0]
        mmap[out : out + n, :] = vecs
        out += n

    bytes_per_vec = dim * 4 + n_neighbors * 8
    n_clusters = max(1, math.ceil((bytes_per_vec * num_rows) / target_gpu_mem)) # build knn graph in stages (1M rows 1 stage, 3M 2 stages)
    print("number of clusters", n_clusters)

    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        build_algo="nn_descent",
        build_kwds={"nnd_do_batch": True, "nnd_n_clusters": n_clusters},
        output_type="numpy",
    )
    # X_low = umap.fit_transform(mmap, data_on_host=True).astype(np.float32)
    ctx = mp.get_context("spawn")
    tmp_out = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp_out.close()

    p = ctx.Process(
        target=_run_umap_worker,
        args=(
            mmap_path,
            (num_rows, dim),
            n_neighbors,
            n_clusters,
            tmp_out.name,
        ),  # we execute UMAP via a child process to avoid python thread heartbeat timeouts
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"UMAP subprocess exited with {p.exitcode}")

    X_low = np.load(tmp_out.name, mmap_mode="r")  # memory-maps the 2-D embedding
    os.unlink(tmp_out.name)

    print("UMAP done")
    if num_rows > 100_000:
        pct_low, pct_high = 0.07, 99.93
    elif num_rows > 10_000:
        pct_low, pct_high = 0.3, 99.7
    elif num_rows > 1_000:
        pct_low, pct_high = 0.5, 99.5
    else:
        pct_low, pct_high = 1.0, 99.0

    pmin, pmax = np.percentile(
        X_low, [pct_low, pct_high], axis=0
    )  # clip data into dimensions where most of the data is

    ranges = pmax - pmin

    X_clip = np.clip(X_low, pmin, pmax)

    X_norm = (X_clip - pmin) / ranges * 100.0
    X_norm = np.clip(X_norm, 0.0, 100.0)

    # jitter the data which was moved to the edges
    eps_jitter = 0.9
    tol_data = 1e-6

    bottom_clip = X_clip[:, 1] <= pmin[1] + tol_data
    top_clip = X_clip[:, 1] >= pmax[1] - tol_data
    left_clip = X_clip[:, 0] <= pmin[0] + tol_data
    right_clip = X_clip[:, 0] >= pmax[0] - tol_data

    X_norm[bottom_clip, 1] = np.random.uniform(0, eps_jitter, size=bottom_clip.sum())
    X_norm[top_clip, 1] = np.random.uniform(100 - eps_jitter, 100, size=top_clip.sum())
    X_norm[left_clip, 0] = np.random.uniform(0, eps_jitter, size=left_clip.sum())
    X_norm[right_clip, 0] = np.random.uniform(
        100 - eps_jitter, 100, size=right_clip.sum()
    )

    idx = np.arange(num_rows, dtype=np.int32)
    perm = np.random.permutation(num_rows)
    x_shuf = X_norm[perm, 0]
    y_shuf = X_norm[perm, 1]
    idx_shuf = idx[perm]

    out_tbl = pa.Table.from_arrays(
        [
            pa.array(x_shuf, type=pa.float32()),
            pa.array(y_shuf, type=pa.float32()),
            pa.array(idx_shuf, type=pa.int32()),  # umap outputs in same order as input
        ],
        names=["x", "y", "row-index"],  # ix == row-index
    )

    xy_path = os.path.join(run_dir, "xy.arrow")
    with pa.OSFile(xy_path, "wb") as sink:
        writer = ipc.new_file(sink, out_tbl.schema)
        writer.write_table(out_tbl)
        writer.close()

    MAX_ATTEMPTS = 5
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            vol.commit()
            break
        except Exception as e:
            print(f"vol.commit() failed on attempt {attempt}/{MAX_ATTEMPTS}: {e!r}")
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    "Could not commit volume after several retries"
                ) from e
            time.sleep(2 ** (attempt - 1))

    del mmap
    os.unlink(mmap_path)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    print("done reducing")


def failed(supabase, projection_id):
    supabase.table("projections").update({"status": "failed"}).eq(
        "projection_id", projection_id
    ).execute()


def _run_umap_worker(
    mmap_path: str,
    shape: tuple[int, int],
    n_neighbors: int,
    n_clusters: int,
    result_path: str,
) -> None:
    """
    Child process entry point.
    Reads the memory-map, runs cuML UMAP, writes a .npy file with (N,2) floats.
    """
    # Everything heavy is imported **inside** the child so that the fork/spawn
    # does not pull GPU state into the parent.
    import numpy as np
    from cuml.manifold import UMAP

    mmap = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=shape)

    umap = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        build_algo="nn_descent",
        build_kwds={"nnd_do_batch": True, "nnd_n_clusters": n_clusters},
        output_type="numpy",
    )

    X_low = umap.fit_transform(mmap, data_on_host=True).astype(np.float32)
    np.save(result_path, X_low)
