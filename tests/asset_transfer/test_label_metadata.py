import json
from pathlib import Path
import tempfile
import unittest


class LabelMetadata(unittest.TestCase):
    def test_labels_follow_frozen_split_without_opening_gt(self):
        from scripts.asset_transfer.select_label_metadata import select_labels
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);splits=root/'splits';labels=root/'labels'
            rows=[dict(dataset='LoveDA',image_id=s,split=s) for s in ('calibration','pilot','confirmation')]
            splits.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            labels.write_text(''.join(json.dumps(dict(dataset='LoveDA',image_id=s,path='/does/not/exist/'+s,sha256='a'*64))+'\n' for s in ('calibration','pilot','confirmation')))
            select_labels(splits,labels,root/'out')
            for split in ('pilot','confirmation'):
                result=[json.loads(x) for x in (root/'out'/f'{split}-labels.jsonl').read_text().splitlines()]
                self.assertEqual([r['image_id'] for r in result],[split])
            with self.assertRaises(FileExistsError):select_labels(splits,labels,root/'out')


if __name__=='__main__':unittest.main()
