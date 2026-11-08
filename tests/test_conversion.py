import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("convert_weights",ROOT / "scripts/convert_weights.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

class Conversion(unittest.TestCase):
    def test_shared_weights_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = converter.Writer(Path(directory) / "weights.bin")
            original = np.arange(16,dtype=np.uint8)
            writer.add("embed",original,"q4")
            writer.add("head",original.copy(),"q4")
            writer.add("other",original + 1,"q4")
            writer.file.close()
            self.assertEqual(writer.index["embed"],writer.index["head"])
            self.assertEqual(writer.index["other"]["offset"],256)
            data = (Path(directory) / "weights.bin").read_bytes()
            self.assertEqual(data[:16],original.tobytes())
            self.assertEqual(data[256:],(original+1).tobytes())

    def test_query_and_gate_head_order(self):
        data=np.arange(8).reshape(8,1)
        result=converter.interleaved_heads_to_split((data,data,data),2,2)
        self.assertEqual(result[0].ravel().tolist(),[0,1,4,5,2,3,6,7])

if __name__=="__main__":unittest.main()
