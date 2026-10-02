# check_schema.py
import pyarrow.parquet as pq

meta = pq.read_table("/groups/icecube/julie/my_converted_data/origin_mc_muongun_130000/test_meta.parquet")
print(meta.schema.names)