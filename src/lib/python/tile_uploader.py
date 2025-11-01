from tile_uploader_utils import upload_to_r2


class TileUploader:

    def __init__(
        self,
        run_dir: str,
        projection_id: str,
        supabase_client,
        arrow_schema,
        timestamp_cols: list[str],
        digest_by_col,
        metadata: dict,
        vector_col: str,
        primary_key_col: str,
        output_dir="/tmp/tiles",
    ):
        import os, glob, pyarrow.dataset as ds

        self.run_dir = run_dir
        self.projection_id = projection_id
        self.supabase_client = supabase_client
        self.arrow_schema = arrow_schema
        self.timestamp_cols = timestamp_cols
        self.digest_by_col = digest_by_col
        self.metadata = metadata
        self.output_dir = output_dir
        self.primary_key_col = primary_key_col
        self.vector_col = vector_col

        data_files = [
            f
            for f in glob.glob(os.path.join(run_dir, "*.arrow"))
            if not f.endswith("xy.arrow")
        ]
        self.dataset = ds.dataset(data_files, format="arrow")
        self.num_rows = self.dataset.count_rows()

    def _upload_tile(self, tile_id, points):  # points is in form: [[x, y, pk],...]
        import pyarrow as pa
        import pyarrow.feather as feather
        import os
        import zstandard as zstd
        from datetime import timezone
        import decimal
        import gc

        print("uploading tile")
        os.makedirs(self.output_dir, exist_ok=True)

        # extract the list of row‐indices
        indices = [int(pt[2]) for pt in points]
        # pull exactly those rows from the dataset
        subset: pa.Table = self.dataset.take(indices)

        if self.vector_col in subset.column_names:
            subset = subset.drop([self.vector_col])

        new_names = []
        for name in subset.column_names:
            if name == self.primary_key_col:
                new_names.append("ix")
            else:
                new_names.append(f"user_{name}")
        subset = subset.rename_columns(new_names)

        # turn into a dict-of‐lists so we can look up row‐i by its position
        data = subset.to_pydict()

        combined = []
        for idx, (x, y, _) in enumerate(points):
            row = {col: data[col][idx] for col in data}
            row["x"] = x
            row["y"] = y
            combined.append(row)

        for row in combined:
            for col, td in self.digest_by_col.items():
                v = row.get(col)
                if v is None:
                    continue
                if isinstance(v, decimal.Decimal):
                    row[col] = float(v)
                if col in self.timestamp_cols:
                    if v.tzinfo is None:
                        v = v.replace(tzinfo=timezone.utc)
                    td.update(v.timestamp() * 1e3)
                else:
                    td.update(float(v))

        print("table ready, tdigest updated")

        out_table = pa.Table.from_pylist(combined, schema=self.arrow_schema)
        uncompressed_filename = os.path.join(
            self.output_dir, f"{tile_id.replace('/', '_')}.arrow"
        )
        feather.write_feather(
            out_table, uncompressed_filename, compression="uncompressed"
        )
        uncompressed_size = os.path.getsize(uncompressed_filename)

        compressed_filename = uncompressed_filename + ".zst"
        cctx = zstd.ZstdCompressor()
        with open(uncompressed_filename, "rb") as f_in, open(
            compressed_filename, "wb"
        ) as f_out:
            f_out.write(cctx.compress(f_in.read()))

        compressed_file_size = os.path.getsize(compressed_filename)

        remote_path = f"{self.projection_id}/tiles/{tile_id}.arrow.zst"
        upload_to_r2(compressed_filename, remote_path)

        tile_metadata = {
            "tile_id": tile_id,
            "uncompressed_size": 0,
            "compressed_size": 0,
            "children": [],
            "node_count": len(points),
        }

        tile_metadata["uncompressed_size"] = uncompressed_size
        tile_metadata["compressed_size"] = compressed_file_size
        self.metadata[tile_id] = tile_metadata
        print("done uploading tile")

        del subset, data, combined, out_table
        gc.collect()

    def recurse(self, qtree, tile_id="0/0_0"):
        """
        Walk the quad‐tree, calling _upload_tile on each leaf, building out
        metadata and digest_by_columns as you go.
        """
        if qtree.nodes:
            pts = qtree.nodes
            self._upload_tile(tile_id, pts)

        if qtree.children:
            z, coords = tile_id.split("/", 1)
            x, y = map(int, coords.split("_"))
            child_ids = [
                f"{int(z)+1}/{2*x}_{2*y}",
                f"{int(z)+1}/{2*x}_{2*y+1}",
                f"{int(z)+1}/{2*x+1}_{2*y}",
                f"{int(z)+1}/{2*x+1}_{2*y+1}",
            ]
            for child, child_id in zip(qtree.children, child_ids):
                self.recurse(child, child_id)
                if child_id in self.metadata:
                    self.metadata[tile_id]["children"].append(self.metadata[child_id])
